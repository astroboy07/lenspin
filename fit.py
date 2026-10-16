from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution, minimize

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
Q_MIN = 0.25
SPIN_MAX = 0.99
LN_B_THRESHOLD = 4.6
SNR_FRACTION_BAND = 0.95
THETA_JN_SCAN = 37
NM_OPTIONS = {"xatol": 1e-4, "fatol": 1e-7, "maxfev": 3000}


@dataclass
class Problem:
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
        mc = chirp_mass(self.m1, self.m2)
        f_min = F_MIN.get(self.network, F_MIN_DEFAULT)
        self.grid = make_grid(f_min, F_MAX, *component_masses(MCHIRP_RANGE[0] * mc, Q_MIN))
        self.net = Network(self.network, self.grid)
        hp, hc = aligned_polarizations(self.m1, self.m2, 0.0, 0.0, np.arccos(self.cosi), self.grid)
        unlensed = self.net.project(hp, hc, self.ra, self.dec, self.psi)
        lens = lens_factor(self.grid.freqs, self.dt, self.flux_ratio)
        self.signal = [lens * h for h in unlensed]
        self.ss = self.net.norm2(self.signal)
        cumulative = self.net.snr_cumulative(unlensed)
        f_band = self.grid.freqs[np.searchsorted(cumulative, SNR_FRACTION_BAND * cumulative[-1])]
        self.n_fringes = self.dt * (f_band - self.grid.f_min)
        self.mchirp = mc
        self.q = self.m2 / self.m1

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

    def ff_aligned(self, x) -> float:
        mchirp, q, chi1z, chi2z = x[:4]
        m1, m2 = component_masses(mchirp, q)
        try:
            h0, _ = aligned_polarizations(m1, m2, chi1z, chi2z, 0.0, self.grid)
        except RuntimeError:
            return 0.0
        ra, dec = self.sky(x)
        return float(np.sqrt(max(aligned_fstat(self.net, self.signal, h0, ra, dec), 0.0) / self.ss))

    def ff_precessing(self, x) -> float:
        try:
            hp, hc = precessing_polarizations(x[:7], self.grid)
        except RuntimeError:
            return 0.0
        ra, dec = self.sky(x)
        return float(np.sqrt(max(precessing_overlap(self.net, self.signal, hp, hc, ra, dec), 0.0) / self.ss))


def optimise(objective, bounds, x0, seed, popsize, maxiter):
    lo, hi = np.array(bounds).T
    x0 = np.clip(x0, lo, hi)
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


def fit_aligned(problem: Problem, seed: int, popsize: int, maxiter: int):
    bounds = [*problem.mass_bounds(), (-SPIN_MAX, SPIN_MAX), (-SPIN_MAX, SPIN_MAX), *problem.sky_bounds()]
    x0 = [problem.mchirp, problem.q, 0.0, 0.0, *problem.sky_truth()]
    return optimise(problem.ff_aligned, bounds, x0, seed, popsize, maxiter)


def precessing_start(problem: Problem, x_aligned) -> list[float]:
    mchirp, q, chi1z, chi2z = x_aligned[:4]
    sky = list(x_aligned[4:])
    base = [mchirp, q, abs(chi1z), 1.0 if chi1z >= 0 else -1.0, chi2z]
    thetas = np.linspace(0.0, np.pi, THETA_JN_SCAN)
    ffs = [problem.ff_precessing([*base, th, 0.0, *sky]) for th in thetas]
    return [*base, float(thetas[int(np.argmax(ffs))]), 0.0, *sky]


def fit_precessing(problem: Problem, x_aligned, seed: int, popsize: int, maxiter: int):
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


def rho_crit(ff: float) -> float:
    return float(np.sqrt(LN_B_THRESHOLD / (1.0 - ff))) if ff < 1.0 else np.inf


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
