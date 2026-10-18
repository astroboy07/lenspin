from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution, minimize, minimize_scalar

from network import F_MAX, F_MIN, F_MIN_DEFAULT, Network, aligned_fstat, make_grid, precessing_overlap
from signals import (
    D_REF_MPC,
    aligned_polarizations,
    chirp_mass,
    component_masses,
    lens_factor,
    precessing_polarizations,
)

MCHIRP_RANGE = (0.85, 1.15)
Q_MIN = 0.1
SPIN_MAX = 0.99
DT_MAX_MS = 60.0
LN_B_THRESHOLD = 4.6
SNR_FRACTION_BAND = 0.95
THETA_JN_SCAN = 37
NM_OPTIONS = {"xatol": 1e-4, "fatol": 1e-7, "maxfev": 3000}
BOUND_MARGIN = 1e-9
SCAN_CHI1 = (0.3, 0.6, 0.9)
SCAN_COS_TILT = (-0.5, 0.0, 0.5)
SCAN_THETA_JN = np.linspace(0.0, np.pi, 12)
SCAN_PHI_JL = np.linspace(0.0, 2 * np.pi, 8, endpoint=False)
SCAN_DT_MS = np.arange(0.25, DT_MAX_MS, 0.25)
SCAN_FLUX = (0.05, 0.15, 0.3, 0.6, 0.9)
TOP_STARTS = 5
LOCAL_OPTIONS = {"xatol": 1e-4, "fatol": 1e-7, "maxfev": 1500}
CROSS_ROUNDS = 2
IMPROVE_TOL = 1e-9


class FitBase:
    def setup(self, m1: float, m2: float, network: str) -> None:
        self.mchirp = chirp_mass(m1, m2)
        self.q = m2 / m1
        f_min = F_MIN.get(network, F_MIN_DEFAULT)
        self.grid = make_grid(f_min, F_MAX, *component_masses(MCHIRP_RANGE[0] * self.mchirp, Q_MIN))
        self.net = Network(network, self.grid)

    def sky(self, x) -> tuple[float, float]:
        if self.sky_free:
            return float(x[-2]), float(np.arcsin(np.clip(x[-1], -1.0, 1.0)))
        return self.ra, self.dec

    def sky_bounds(self) -> list[tuple[float, float]]:
        return [(0.0, 2 * np.pi), (-1.0, 1.0)] if self.sky_free else []

    def sky_truth(self) -> list[float]:
        return [self.ra, np.sin(self.dec)] if self.sky_free else []

    def mass_bounds(self) -> list[tuple[float, float]]:
        return [(MCHIRP_RANGE[0] * self.mchirp, MCHIRP_RANGE[1] * self.mchirp), (Q_MIN, 1.0)]

    def face_on(self, x):
        mchirp, q, chi1z, chi2z = x[:4]
        m1, m2 = component_masses(mchirp, q)
        return aligned_polarizations(m1, m2, chi1z, chi2z, 0.0, self.grid)[0]

    def fstat_ff(self, h0, x) -> float:
        ra, dec = self.sky(x)
        return float(np.sqrt(max(aligned_fstat(self.net, self.signal, h0, ra, dec), 0.0) / self.ss))

    def ff_aligned(self, x) -> float:
        try:
            h0 = self.face_on(x)
        except RuntimeError:
            return 0.0
        return self.fstat_ff(h0, x)

    def ff_lensed(self, x) -> float:
        try:
            h0 = self.face_on(x)
        except RuntimeError:
            return 0.0
        return self.fstat_ff(h0 * lens_factor(self.grid.freqs, x[4] * 1e-3, x[5]), x)

    def ff_precessing(self, x) -> float:
        try:
            hp, hc = precessing_polarizations(x[:7], self.grid)
        except RuntimeError:
            return 0.0
        ra, dec = self.sky(x)
        return float(np.sqrt(max(precessing_overlap(self.net, self.signal, hp, hc, ra, dec), 0.0) / self.ss))


@dataclass
class Problem(FitBase):
    m1: float
    m2: float
    dt: float
    flux_ratio: float
    cosi: float
    psi: float
    ra: float
    dec: float
    network: str
    sky_free: bool

    def __post_init__(self) -> None:
        self.setup(self.m1, self.m2, self.network)
        hp, hc = aligned_polarizations(self.m1, self.m2, 0.0, 0.0, np.arccos(self.cosi), self.grid)
        unlensed = self.net.project(hp, hc, self.ra, self.dec, self.psi)
        lens = lens_factor(self.grid.freqs, self.dt, self.flux_ratio)
        self.signal = [lens * h for h in unlensed]
        self.ss = self.net.norm2(self.signal)
        cumulative = self.net.snr_cumulative(unlensed)
        f_band = self.grid.freqs[np.searchsorted(cumulative, SNR_FRACTION_BAND * cumulative[-1])]
        self.n_fringes = self.dt * (f_band - self.grid.f_min)


@dataclass
class PrecessingProblem(FitBase):
    m1: float
    m2: float
    chi1: float
    cos_tilt1: float
    chi2z: float
    theta_jn: float
    phi_jl: float
    psi: float
    ra: float
    dec: float
    network: str
    sky_free: bool

    def __post_init__(self) -> None:
        self.setup(self.m1, self.m2, self.network)
        x = [self.mchirp, self.q, self.chi1, self.cos_tilt1, self.chi2z, self.theta_jn, self.phi_jl]
        hp, hc = precessing_polarizations(x, self.grid)
        self.signal = self.net.project(hp, hc, self.ra, self.dec, self.psi)
        self.ss = self.net.norm2(self.signal)
        self.chi_perp = self.chi1 * np.sqrt(1.0 - self.cos_tilt1**2)
        cumulative = self.net.snr_cumulative(self.signal)
        self.f_band = self.grid.freqs[np.searchsorted(cumulative, SNR_FRACTION_BAND * cumulative[-1])]


def optimise(objective, bounds, x0, seed, popsize, maxiter):
    lo, hi = np.array(bounds).T
    # differential_evolution rescales x0 and rejects a value on a bound that rounds to 1 + 1e-16.
    margin = BOUND_MARGIN * (hi - lo)
    x0 = np.clip(x0, lo + margin, hi - margin)
    de = differential_evolution(
        lambda x: 1.0 - objective(x),
        bounds,
        x0=x0,
        seed=seed,
        popsize=popsize,
        maxiter=maxiter,
        tol=1e-8,
        init="sobol",
        polish=False,
    )
    nm = minimize(lambda x: 1.0 - objective(x), de.x, method="Nelder-Mead", bounds=bounds, options=NM_OPTIONS)
    best = nm if nm.fun < de.fun else de
    return 1.0 - float(best.fun), np.asarray(best.x), int(de.nfev + nm.nfev)


def fit_aligned(problem: FitBase, seed: int, popsize: int, maxiter: int):
    bounds = aligned_bounds(problem)
    x0 = [problem.mchirp, problem.q, 0.0, 0.0, *problem.sky_truth()]
    return optimise(problem.ff_aligned, bounds, x0, seed, popsize, maxiter)


def precessing_start(problem: FitBase, x_aligned) -> list[float]:
    mchirp, q, chi1z, chi2z = x_aligned[:4]
    sky = list(x_aligned[4:])
    base = [mchirp, q, abs(chi1z), 1.0 if chi1z >= 0 else -1.0, chi2z]
    thetas = np.linspace(0.0, np.pi, THETA_JN_SCAN)
    ffs = [problem.ff_precessing([*base, th, 0.0, *sky]) for th in thetas]
    step = thetas[1] - thetas[0]
    best = float(thetas[int(np.argmax(ffs))])
    res = minimize_scalar(
        lambda th: -problem.ff_precessing([*base, th, 0.0, *sky]),
        bounds=(max(best - step, 0.0), min(best + step, np.pi)),
        method="bounded",
        options={"xatol": 1e-5},
    )
    theta = float(res.x) if -res.fun > max(ffs) else best
    return [*base, theta, 0.0, *sky]


def fit_precessing(problem: FitBase, x_aligned, seed: int, popsize: int, maxiter: int):
    bounds = [
        *problem.mass_bounds(),
        (0.0, SPIN_MAX),
        (-1.0, 1.0),
        (-SPIN_MAX, SPIN_MAX),
        (0.0, np.pi),
        (0.0, 2 * np.pi),
        *problem.sky_bounds(),
    ]
    return optimise(
        problem.ff_precessing, bounds, precessing_start(problem, x_aligned), seed, popsize, maxiter
    )


def fit_lensed(problem: FitBase, x_aligned, seed: int, popsize: int, maxiter: int):
    bounds = [
        *problem.mass_bounds(),
        (-SPIN_MAX, SPIN_MAX),
        (-SPIN_MAX, SPIN_MAX),
        (0.0, DT_MAX_MS),
        (0.0, 1.0),
        *problem.sky_bounds(),
    ]
    x0 = [*x_aligned[:4], 0.5 * DT_MAX_MS, 0.0, *x_aligned[4:]]
    return optimise(problem.ff_lensed, bounds, x0, seed, popsize, maxiter)


def rho_crit(ff: float) -> float:
    return float(np.sqrt(LN_B_THRESHOLD / (1.0 - ff))) if ff < 1.0 else np.inf


def rho_from_gain(ff_better: float, ff_worse: float) -> float:
    gain = ff_better**2 - ff_worse**2
    return float(np.sqrt(2 * LN_B_THRESHOLD / gain)) if gain > 0 else np.inf


def run_problem(problem: Problem, seed: int, popsize: int, maxiter: int) -> dict:
    ff_al, x_al, nfev_al = fit_aligned(problem, seed, popsize, maxiter)
    ff_pr, x_pr, nfev_pr = fit_precessing(problem, x_al, seed, popsize, maxiter)
    snr = float(np.sqrt(problem.ss))
    rc = rho_crit(ff_pr)
    return {
        "n_fringes": problem.n_fringes,
        "snr_dref": snr,
        "ff_aligned": ff_al,
        "ff_precessing": ff_pr,
        "rho_crit_aligned": rho_crit(ff_al),
        "rho_crit_precessing": rc,
        "d_crit_mpc": D_REF_MPC * snr / rc,
        "nfev": nfev_al + nfev_pr,
        "best_aligned": " ".join(f"{v:.5g}" for v in x_al),
        "best_precessing": " ".join(f"{v:.5g}" for v in x_pr),
    }


def run_reverse_problem(problem: PrecessingProblem, seed: int, popsize: int, maxiter: int) -> dict:
    ff_al, x_al, nfev_al = fit_aligned(problem, seed, popsize, maxiter)
    ff_le, x_le, nfev_le = fit_lensed(problem, x_al, seed, popsize, maxiter)
    return {
        "chi_perp": problem.chi_perp,
        "snr_dref": float(np.sqrt(problem.ss)),
        "ff_aligned": ff_al,
        "ff_lensed": ff_le,
        "rho_false_lensing": rho_from_gain(ff_le, ff_al),
        "rho_crit_lensing": rho_from_gain(1.0, ff_le),
        "best_dt_ms": float(x_le[4]),
        "best_flux_ratio": float(x_le[5]),
        "nfev": nfev_al + nfev_le,
        "best_aligned": " ".join(f"{v:.5g}" for v in x_al),
        "best_lensed": " ".join(f"{v:.5g}" for v in x_le),
    }


def precessing_bounds(problem: FitBase) -> list[tuple[float, float]]:
    return [
        *problem.mass_bounds(),
        (0.0, SPIN_MAX),
        (-1.0, 1.0),
        (-SPIN_MAX, SPIN_MAX),
        (0.0, np.pi),
        (0.0, 2 * np.pi),
        *problem.sky_bounds(),
    ]


def lensed_bounds(problem: FitBase) -> list[tuple[float, float]]:
    return [
        *problem.mass_bounds(),
        (-SPIN_MAX, SPIN_MAX),
        (-SPIN_MAX, SPIN_MAX),
        (0.0, DT_MAX_MS),
        (0.0, 1.0),
        *problem.sky_bounds(),
    ]


def objective_bounds(problem: FitBase, kind: str):
    if kind == "precessing":
        return problem.ff_precessing, precessing_bounds(problem)
    return problem.ff_lensed, lensed_bounds(problem)


def family(problem: FitBase, kind: str, x_aligned):
    mchirp, q, chi1z, chi2z = x_aligned[:4]
    sky = list(x_aligned[4:])
    if kind == "precessing":
        scan = [
            [mchirp, q, chi1, cos_tilt, chi2z, theta, phi, *sky]
            for chi1 in SCAN_CHI1
            for cos_tilt in SCAN_COS_TILT
            for theta in SCAN_THETA_JN
            for phi in SCAN_PHI_JL
        ]
        start = precessing_start(problem, x_aligned)
        return problem.ff_precessing, precessing_bounds(problem), start, scan
    scan = [[mchirp, q, chi1z, chi2z, dt, flux, *sky] for dt in SCAN_DT_MS for flux in SCAN_FLUX]
    start = [mchirp, q, chi1z, chi2z, 0.5 * DT_MAX_MS, 0.0, *sky]
    return problem.ff_lensed, lensed_bounds(problem), start, scan


def polish(objective, bounds, x0):
    lo, hi = np.array(bounds).T
    res = minimize(
        lambda x: 1.0 - objective(x),
        np.clip(x0, lo, hi),
        method="Nelder-Mead",
        bounds=bounds,
        options=LOCAL_OPTIONS,
    )
    return 1.0 - float(res.fun), np.asarray(res.x), int(res.nfev)


def multi_start(objective, bounds, start, scan):
    scores = np.array([objective(x) for x in scan])
    tops = [scan[i] for i in np.argsort(scores)[::-1][:TOP_STARTS]]
    best, nfev = (-1.0, None), len(scan)
    for x0 in [start, *tops]:
        ff, x, n = polish(objective, bounds, x0)
        nfev += n
        if ff > best[0]:
            best = (ff, x)
    return best[0], best[1], nfev


def run_joint(problems: dict, kind: str, seed: int, popsize: int, maxiter: int) -> dict:
    fits = {}
    for net, p in problems.items():
        ff_al, x_al, n_al = fit_aligned(p, seed, popsize, maxiter)
        objective, bounds, start, scan = family(p, kind, x_al)
        ff_de, x_de, n_de = optimise(objective, bounds, start, seed, popsize, maxiter)
        ff_ms, x_ms, n_ms = multi_start(objective, bounds, start, scan)
        own = (ff_de, x_de) if ff_de >= ff_ms else (ff_ms, x_ms)
        fits[net] = {
            "aligned": (ff_al, x_al),
            "aligned_own": ff_al,
            "best": own,
            "own": own[0],
            "de": ff_de,
            "multistart": ff_ms,
            "best_from": "own",
            "nfev": n_al + n_de + n_ms,
        }
    for _ in range(CROSS_ROUNDS):
        for net, p in problems.items():
            f = fits[net]
            for other, g in fits.items():
                if other == net:
                    continue
                ff, x, n = polish(p.ff_aligned, aligned_bounds(p), g["aligned"][1])
                f["nfev"] += n
                if ff > f["aligned"][0] + IMPROVE_TOL:
                    f["aligned"] = (ff, x)
                objective, bounds = objective_bounds(p, kind)
                ff, x, n = polish(objective, bounds, g["best"][1])
                f["nfev"] += n
                if ff > f["best"][0] + IMPROVE_TOL:
                    f["best"], f["best_from"] = (ff, x), other
    for net, p in problems.items():
        f = fits[net]
        if f["aligned"][0] > f["best"][0]:
            objective, bounds, start, _ = family(p, kind, f["aligned"][1])
            ff, x, n = polish(objective, bounds, start)
            f["nfev"] += n
            if ff > f["best"][0]:
                f["best"], f["best_from"] = (ff, x), "aligned"
    return fits


def aligned_bounds(problem: FitBase) -> list[tuple[float, float]]:
    return [*problem.mass_bounds(), (-SPIN_MAX, SPIN_MAX), (-SPIN_MAX, SPIN_MAX), *problem.sky_bounds()]


def fmt(x) -> str:
    return " ".join(f"{v:.5g}" for v in x)


def joint_forward_rows(problems: dict, seed: int, popsize: int, maxiter: int) -> dict:
    fits = run_joint(problems, "precessing", seed, popsize, maxiter)
    rows = {}
    for net, p in problems.items():
        f = fits[net]
        ff_al, ff_pr = f["aligned"][0], f["best"][0]
        snr = float(np.sqrt(p.ss))
        rc = rho_crit(ff_pr)
        rows[net] = {
            "n_fringes": p.n_fringes,
            "snr_dref": snr,
            "ff_aligned": ff_al,
            "ff_precessing": ff_pr,
            "ff_aligned_own": f["aligned_own"],
            "ff_precessing_own": f["own"],
            "ff_precessing_de": f["de"],
            "ff_precessing_multistart": f["multistart"],
            "best_from": f["best_from"],
            "rho_crit_aligned": rho_crit(ff_al),
            "rho_crit_precessing": rc,
            "d_crit_mpc": D_REF_MPC * snr / rc,
            "nfev": f["nfev"],
            "best_aligned": fmt(f["aligned"][1]),
            "best_precessing": fmt(f["best"][1]),
        }
    return rows


def joint_reverse_rows(problems: dict, seed: int, popsize: int, maxiter: int) -> dict:
    fits = run_joint(problems, "lensed", seed, popsize, maxiter)
    rows = {}
    for net, p in problems.items():
        f = fits[net]
        ff_al, ff_le = f["aligned"][0], f["best"][0]
        x_le = f["best"][1]
        rows[net] = {
            "chi_perp": p.chi_perp,
            "snr_dref": float(np.sqrt(p.ss)),
            "ff_aligned": ff_al,
            "ff_lensed": ff_le,
            "ff_aligned_own": f["aligned_own"],
            "ff_lensed_own": f["own"],
            "ff_lensed_de": f["de"],
            "ff_lensed_multistart": f["multistart"],
            "best_from": f["best_from"],
            "rho_false_lensing": rho_from_gain(ff_le, ff_al),
            "rho_crit_lensing": rho_from_gain(1.0, ff_le),
            "best_dt_ms": float(x_le[4]),
            "best_flux_ratio": float(x_le[5]),
            "best_n_fringes": float(x_le[4]) * 1e-3 * (p.f_band - p.grid.f_min),
            "nfev": f["nfev"],
            "best_aligned": fmt(f["aligned"][1]),
            "best_lensed": fmt(x_le),
        }
    return rows
