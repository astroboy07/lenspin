import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from pathlib import Path

import bilby
import lal
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from network import F_MAX, F_MIN_DEFAULT, PSD_MODELS, UPSAMPLE, FrequencyGrid, noise_weights
from pe import DT_MAX, Q_MIN, SPIN_MAX, STUDIES, make_ifos, segment_duration
from signals import aligned_polarizations, chirp_mass, component_masses, lens_factor
from waveoptics import wave_lens_factor

SNR = 25.0
NAMES = ("log_mchirp", "eta", "chi", "tc", "phic", "log_amp", "lens_dt", "lens_flux")
STEPS = {0: 1e-5, 1: 1e-5, 2: 1e-4, 6: 1e-6, 7: 1e-5}
U_STEP = 1e-4
ETA_MAX = 0.25
ETA_MIN = Q_MIN / (1 + Q_MIN) ** 2
CHI_MAX = 0.99
FLUX_MIN = 1e-6
FIT_ITER = 400
FIT_TOL = 1e-9
LENSINGS = ("full", "amplitude", "phase")
SCAN_DT = np.linspace(0.0, 0.05, 501)
SCAN_FLUX = 1e-6
NMCHIRP = 81
START_Q = (0.1, 0.2, 0.35, 0.55, 0.8, 1.0)
START_CHI = (-0.7, -0.35, 0.0, 0.35, 0.7)
RIDGE_DLNL = 0.5
TOY = {"m1": 10.0, "m2": 8.0, "duration": 16.0, "mchirp_span": (0.9, 1.1)}
TOY_SETS = {"full": (0, 1, 2, 3, 4, 5), "fix_chi": (0, 1, 3, 4, 5), "fix_eta": (0, 2, 3, 4, 5)}
ALL = (0, 1, 2, 3, 4, 5)
AT_FIXED_Q = (0, 2, 3, 4, 5)
LENSED_AT_FIXED_Q = (0, 2, 3, 4, 5, 6, 7)
PROFILE_Q = np.unique(np.concatenate([np.geomspace(Q_MIN, 0.5, 14), np.linspace(0.5, 1.0, 15)]))
PROFILE_STARTS = 6
CONTINUATION_GAP = 5.0
CONTINUATION_SWEEPS = 3
CONTINUATION_STEP = 0.02
SAME_MODE_D2 = 4.0
POSTERIOR_SAMPLES = 40_000
LOW_Q_MODE = {"mchirp": 62.0, "q": 0.5, "chi_eff": -0.4}
IS_SAMPLES = 8000
IS_INFLATE = 1.3
U_FLAT = 0.5
U_CELLS = 4000


def eta_to_q(eta: float) -> float:
    eta = min(eta, ETA_MAX)
    return (1.0 - 2.0 * eta - np.sqrt(max(1.0 - 4.0 * eta, 0.0))) / (2.0 * eta)


def lensing(freqs: np.ndarray, dt: float, flux: float, kind: str) -> np.ndarray:
    factor = lens_factor(freqs, dt, flux)
    if kind == "amplitude":
        return np.abs(factor)
    if kind == "phase":
        return factor / np.abs(factor)
    return factor


def xp_shape(grid: FrequencyGrid, mchirp: float, eta: float, chi1: float, chi2: float) -> np.ndarray:
    m1, m2 = component_masses(mchirp, eta_to_q(eta))
    return aligned_polarizations(m1, m2, chi1, chi2, 0.0, grid)[0]


def taylorf2_shape(grid: FrequencyGrid, mchirp: float, eta: float, chi1: float, chi2: float,
                   f_cut: float) -> np.ndarray:  # fmt: skip
    """Newtonian amplitude, 2PN phase with the 1.5PN spin-orbit term (equal aligned spins)."""
    f = grid.freqs
    v = (np.pi * mchirp * eta**-0.6 * lal.MTSUN_SI * f) ** (1.0 / 3.0)
    beta = 0.5 * (chi1 + chi2) * (113.0 - 76.0 * eta) / 12.0
    psi = 3.0 / (128.0 * eta * v**5) * (
        1.0
        + (3715.0 / 756.0 + 55.0 * eta / 9.0) * v**2
        + (4.0 * beta - 16.0 * np.pi) * v**3
        + (15293365.0 / 508032.0 + 27145.0 * eta / 504.0 + 3085.0 * eta**2 / 72.0) * v**4
    )
    return np.where(f <= f_cut, mchirp ** (5.0 / 6.0) * f ** (-7.0 / 6.0) * np.exp(-1j * psi), 0.0)


def log_spin_prior(chi: np.ndarray) -> np.ndarray:
    """AlignedSpin prior with a ~ U(0, SPIN_MAX): p(chi) = ln(SPIN_MAX / |chi|) / (2 SPIN_MAX)."""
    chi = np.abs(np.asarray(chi, float))
    inside = (chi > 0) & (chi < SPIN_MAX)
    safe = np.where(inside, chi, 0.5 * SPIN_MAX)
    return np.where(inside, np.log(np.log(SPIN_MAX / safe) / (2 * SPIN_MAX)), -np.inf)


def u_support(chi: float, q: float) -> tuple[float, float]:
    """Range of u with chi1 = chi + q u and chi2 = chi - u inside the spin prior."""
    return max((-SPIN_MAX - chi) / q, chi - SPIN_MAX), min((SPIN_MAX - chi) / q, chi + SPIN_MAX)


def log_spin_integral(chi: float, q: float, grad: float, curv: float) -> float:
    """log of the integral over u of p(chi1) p(chi2) (1 + q) exp(grad u - curv u^2 / 2)."""
    lo, hi = u_support(chi, q)
    peak = np.clip(grad / curv, lo, hi) if curv > 0 else (hi if grad > 0 else lo)
    top = grad * peak - 0.5 * curv * peak**2
    integrand = lambda u: np.exp(
        log_spin_prior(chi + q * u) + log_spin_prior(chi - u) + grad * u - 0.5 * curv * u * u - top
    )
    points = sorted({p for p in (-chi / q, chi, peak) if lo < p < hi})
    value = quad(integrand, lo, hi, points=points, limit=400)[0]
    return float(np.log(max(value, 1e-300)) + top + np.log1p(q))


def u_proposal(chi: float, q: float, grad: float, curv: float, rng: np.random.Generator) -> tuple:
    """Draws of u from a piecewise-constant density following p(chi1) p(chi2) exp(grad u - curv u^2 / 2)."""
    lo, hi = u_support(chi, q)
    edges = np.linspace(lo, hi, U_CELLS + 1)
    mid = 0.5 * (edges[1:] + edges[:-1])
    log_f = log_spin_prior(chi + q * mid) + log_spin_prior(chi - mid) + grad * mid - 0.5 * curv * mid**2
    finite = np.isfinite(log_f)
    mass = np.where(finite, np.exp(log_f - log_f[finite].max()), 0.0) if finite.any() else np.ones(U_CELLS)
    mass /= mass.sum()
    cell = rng.choice(U_CELLS, IS_SAMPLES, p=mass)
    width = edges[1] - edges[0]
    return edges[cell] + width * rng.uniform(size=IS_SAMPLES), np.log(mass[cell] / width)


def log_z_from(terms: dict, snr_scale: float = 1.0) -> float:
    """Fixed-q Laplace log evidence at SNR scaled by snr_scale (same parameters, data x snr_scale).

    The common log_amp shift (-3 ln snr_scale) is dropped: it cancels between solutions and models.
    """
    s2 = snr_scale**2
    k = terms["nfree"]
    return float(
        s2 * terms["loglike"] + 0.5 * k * np.log(2 * np.pi) - 0.5 * (terms["logdet"] + k * np.log(s2))
        + log_spin_integral(terms["chi_eff"], terms["q"], s2 * terms["grad"], s2 * terms["curv"]) + terms["prior"]
    )  # fmt: skip


def log_prior_q(q: np.ndarray) -> np.ndarray:
    """UniformInComponentsMassRatio."""
    return 0.4 * np.log1p(q) - 1.2 * np.log(q)


def log_prior(lam: np.ndarray) -> float:
    """PE priors other than spins and q (those are integrated separately).

    Chirp mass p(Mc) ~ Mc, so p(log Mc) ~ Mc^2. Distance p(D) ~ D^2 and D_eff ~ e^{-log_amp},
    so p(log_amp) ~ e^{-3 log_amp}. Time and phase are uniform. Lens delay U(0, DT_MAX), flux U(0, 1).
    """
    return 2.0 * lam[0] - 3.0 * lam[5] - (np.log(DT_MAX) if len(lam) == 8 else 0.0)


class Model:
    """Template A e^{i phic - 2 pi i f tc} h(Mc, eta, chi1 = chi + q u, chi2 = chi - u) [x F(dt, flux)].

    Exact for an aligned 22-only signal in any network with one PSD shape: the extrinsics reduce to one
    complex amplitude and a time. The data are the lensed injection normalised to the target SNR.
    """

    def __init__(self, inj: dict, approximant: str = "xp", snr: float = SNR, optics: str = "geometric") -> None:
        df = 1.0 / segment_duration(inj)
        self.grid = FrequencyGrid(df, round(F_MIN_DEFAULT / df), round(F_MAX / df))
        self.f = self.grid.freqs
        self.weight = noise_weights(PSD_MODELS["H1"], self.grid)
        mtotal = inj["m1"] + inj["m2"]
        f_isco = 1.0 / (6.0**1.5 * np.pi * mtotal * lal.MTSUN_SI)
        if approximant == "taylorf2":
            self.shape = lambda mc, eta, c1, c2: taylorf2_shape(self.grid, mc, eta, c1, c2, f_isco)
        else:
            self.shape = lambda mc, eta, c1, c2: xp_shape(self.grid, mc, eta, c1, c2)
        q = inj["m2"] / inj["m1"]
        self.truth = np.array([np.log(chirp_mass(inj["m1"], inj["m2"])), q / (1.0 + q) ** 2, inj.get("chi_eff", 0.0),
                               0, 0, 0], float)  # fmt: skip
        self.log_mchirp_range = self.truth[0] + np.log(inj["mchirp_span"])
        self.lens_truth = (inj.get("dt", 0.0), inj.get("flux", 0.0))
        if self.lens_truth[1] == 0:
            self.lens = np.ones_like(self.f, complex)
        elif optics == "wave":
            self.lens = wave_lens_factor(self.f, *self.lens_truth)
        else:
            self.lens = lens_factor(self.f, *self.lens_truth)
        self.scale = 1.0
        lensed = self.template(self.truth) * self.lens
        self.scale = snr / np.sqrt(self.inner(lensed, lensed))
        self.h0 = self.template(self.truth)
        self.lensed = self.h0 * self.lens

    def inner(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.real(np.sum(np.conj(a) * b * self.weight, axis=-1))

    def template(self, lam: np.ndarray, u: float = 0.0) -> np.ndarray:
        lmc, eta, chi, tc, phic, lamp = lam[:6]
        q = eta_to_q(eta)
        h = self.scale * np.exp(lamp + 1j * phic - 2j * np.pi * self.f * tc) * self.shape(
            np.exp(lmc), eta, chi + q * u, chi - u
        )
        return h * lens_factor(self.f, lam[6], lam[7]) if len(lam) == 8 else h

    def partial(self, lam: np.ndarray, i: int, step_scale: float = 1.0) -> np.ndarray:
        step = STEPS[i] * step_scale
        e = np.zeros(len(lam))
        e[i] = step
        if i == 1 and lam[1] + step > ETA_MAX or i in (6, 7) and lam[i] - step < 0:
            sign = -1 if i == 1 else 1
            return sign * (-3 * self.template(lam) + 4 * self.template(lam + sign * e)
                           - self.template(lam + 2 * sign * e)) / (2 * step)  # fmt: skip
        return (self.template(lam + e) - self.template(lam - e)) / (2 * step)

    def partial_u(self, lam: np.ndarray) -> np.ndarray:
        return (self.template(lam, U_STEP) - self.template(lam, -U_STEP)) / (2 * U_STEP)

    def derivatives(self, lam: np.ndarray, step_scale: float = 1.0, free: tuple = ALL) -> tuple[np.ndarray, np.ndarray]:
        h = self.template(lam)
        extrinsic = {3: -2j * np.pi * self.f * h, 4: 1j * h, 5: h}
        d = [extrinsic[i] if i in extrinsic else self.partial(lam, i, step_scale) for i in free]
        return h, np.array(d)

    def fisher(self, d: np.ndarray) -> np.ndarray:
        return self.inner(d[:, None, :], d[None, :, :])

    def data(self, dt: float, flux: float, kind: str = "full") -> np.ndarray:
        return self.h0 * lensing(self.f, dt, flux, kind)

    def linear_bias(self, data: np.ndarray, step_scale: float = 1.0) -> np.ndarray:
        _, d = self.derivatives(self.truth, step_scale)
        return np.linalg.solve(self.fisher(d), self.inner(d, data - self.h0))

    def loglike(self, data: np.ndarray, lam: np.ndarray, u: float = 0.0) -> float:
        h = self.template(lam, u)
        return float(self.inner(data, h) - 0.5 * self.inner(h, h))

    def clip(self, lam: np.ndarray) -> np.ndarray:
        lam = lam.copy()
        lam[0] = np.clip(lam[0], *self.log_mchirp_range)
        lam[1] = np.clip(lam[1], ETA_MIN, ETA_MAX)
        lam[2] = np.clip(lam[2], -CHI_MAX, CHI_MAX)
        if len(lam) == 8:
            lam[6] = np.clip(lam[6], 0.0, DT_MAX)
            lam[7] = np.clip(lam[7], FLUX_MIN, 1.0)
        return lam

    def fit(self, data: np.ndarray, start: np.ndarray, free: tuple = ALL) -> tuple[np.ndarray, float]:
        """Levenberg-Marquardt maximisation of the likelihood over the free parameters."""
        lam, mu = self.clip(start), 1e-3
        best = self.loglike(data, lam)
        for _ in range(FIT_ITER):
            h, d = self.derivatives(lam, free=free)
            gamma = self.fisher(d)
            step = np.linalg.solve(gamma + mu * np.diag(np.diag(gamma)), self.inner(d, data - h))
            trial = lam.copy()
            trial[list(free)] += step
            trial = self.clip(trial)
            try:
                value = self.loglike(data, trial)
            except RuntimeError:
                value = -np.inf
            if value > best:
                gain = value - best
                lam, best, mu = trial, value, max(mu / 3.0, 1e-9)
                if gain < FIT_TOL:
                    break
            else:
                mu *= 4.0
                if mu > 1e8:
                    break
        return lam, best

    def peak(self, data: np.ndarray, s: np.ndarray, refine: bool = True) -> tuple[float, complex]:
        """Time and complex overlap Z(t) = sum conj(s) d w e^{2 pi i f t} at the peak of |Z|."""
        c = np.zeros(UPSAMPLE * 2 ** int(np.ceil(np.log2(self.grid.k_max + 1))), complex)
        c[self.grid.k_min : self.grid.k_max + 1] = np.conj(s) * data * self.weight
        z = np.fft.ifft(c) * len(c)
        dt_step = 1.0 / (len(c) * self.grid.df)
        j = int(np.argmax(np.abs(z)))
        t = (j if j < len(c) // 2 else j - len(c)) * dt_step
        if not refine:
            return t, z[j]
        overlap = lambda t: np.sum(np.conj(s) * data * self.weight * np.exp(2j * np.pi * self.f * t))
        res = minimize_scalar(lambda t: -abs(overlap(t)), bounds=(t - dt_step, t + dt_step), method="bounded")
        return (res.x, overlap(res.x)) if -res.fun > abs(z[j]) else (t, z[j])

    def fitting_factor(self, data: np.ndarray, mchirp: float, eta: float, chi1: float, chi2: float,
                       refine: bool = True) -> float:  # fmt: skip
        """Match of a fixed intrinsic template, maximised over time, phase and amplitude."""
        s = self.scale * self.shape(mchirp, eta, chi1, chi2)
        _, z = self.peak(data, s, refine)
        return float(abs(z) / np.sqrt(self.inner(s, s) * self.inner(data, data)))

    def align(self, data: np.ndarray, lam: np.ndarray) -> np.ndarray:
        """Set tc, phic and log_amp to their best values for the rest of lam."""
        base = lam.copy()
        base[3:6] = 0.0
        s = self.template(base)
        t, z = self.peak(data, s)
        out = base.copy()
        out[3:6] = t, np.angle(z), np.log(abs(z) / self.inner(s, s))
        return out

    def best_log_mchirp(self, data: np.ndarray, eta: float, chi: float) -> float:
        """Chirp mass maximising the match at fixed eta and chi, over the PE prior range."""
        grid = np.linspace(*self.log_mchirp_range, NMCHIRP)
        match = lambda lmc: self.fitting_factor(data, np.exp(lmc), eta, chi, chi, refine=False)
        values = [match(x) for x in grid]
        j = int(np.argmax(values))
        lo, hi = grid[max(j - 1, 0)], grid[min(j + 1, NMCHIRP - 1)]
        res = minimize_scalar(lambda x: -match(x), bounds=(lo, hi), method="bounded")
        return float(res.x) if -res.fun > values[j] else float(grid[j])

    def spin_direction(self, data: np.ndarray, lam: np.ndarray, d: np.ndarray, gamma: np.ndarray) -> tuple:
        """Gradient and curvature of the log-likelihood along u, with the free parameters re-optimised."""
        resid = data - self.template(lam)
        du = self.partial_u(lam)
        cross = self.inner(d, du)
        solve = np.linalg.solve(gamma, cross)
        curv = float(self.inner(du, du) - cross @ solve)
        grad = float(self.inner(du, resid) - solve @ self.inner(d, resid))
        return grad, curv, cross, du

    def laplace_terms(self, data: np.ndarray, lam: np.ndarray, free: tuple = AT_FIXED_Q) -> dict:
        """Pieces of the fixed-q Laplace evidence. loglike, logdet, grad and curv scale as SNR^2."""
        h, d = self.derivatives(lam, free=free)
        gamma = self.fisher(d)
        grad, curv, _, _ = self.spin_direction(data, lam, d, gamma)
        cov = np.linalg.inv(gamma)
        return {
            "q": eta_to_q(lam[1]), "chi_eff": float(lam[2]), "nfree": len(free),
            "loglike": float(self.inner(data, h) - 0.5 * self.inner(h, h)),
            "logdet": float(np.linalg.slogdet(gamma)[1]), "grad": grad, "curv": curv, "prior": log_prior(lam),
            "var_log_mchirp": cov[0, 0], "var_chi": cov[1, 1], "cov_log_mchirp_chi": cov[0, 1],
        }  # fmt: skip

    def importance(self, data: np.ndarray, lam: np.ndarray, rng: np.random.Generator) -> dict:
        """Importance-sampled log evidence at fixed q over (log_mchirp, chi, tc, phic, log_amp, u)."""
        q = eta_to_q(lam[1])
        _, d = self.derivatives(lam, free=AT_FIXED_Q)
        gamma = self.fisher(d)
        grad, curv, cross, _ = self.spin_direction(data, lam, d, gamma)
        lo, hi = u_support(lam[2], q)
        if hi - lo < 1e-6:
            return {"log_z_is": -np.inf, "ess": 0.0}
        y0 = lam[list(AT_FIXED_Q)]
        if curv > 0 and 1 / np.sqrt(curv) < U_FLAT:
            u0 = float(np.clip(grad / curv, lo, hi))
            full = np.block([[gamma, cross[:, None]], [cross[None, :], np.array([[curv + cross @ np.linalg.solve(gamma, cross)]])]])
            mean = np.append(y0 - np.linalg.solve(gamma, cross) * u0, u0)
            cov = IS_INFLATE**2 * np.linalg.inv(full)
            draws = rng.multivariate_normal(mean, cov, IS_SAMPLES)
            diff = draws - mean
            log_g = -0.5 * np.einsum("ni,ij,nj->n", diff, np.linalg.inv(cov), diff) - 0.5 * np.linalg.slogdet(2 * np.pi * cov)[1]
        else:
            cov = IS_INFLATE**2 * np.linalg.inv(gamma)
            ys = rng.multivariate_normal(y0, cov, IS_SAMPLES)
            us, log_gu = u_proposal(lam[2], q, grad, curv, rng)
            draws = np.column_stack([ys, us])
            diff = ys - y0
            log_g = (-0.5 * np.einsum("ni,ij,nj->n", diff, np.linalg.inv(cov), diff)
                     - 0.5 * np.linalg.slogdet(2 * np.pi * cov)[1] + log_gu)  # fmt: skip
        log_w = np.full(IS_SAMPLES, -np.inf)
        for n, row in enumerate(draws):
            sample = lam.copy()
            sample[list(AT_FIXED_Q)] = row[:5]
            u = row[5]
            prior = log_spin_prior(sample[2] + q * u) + log_spin_prior(sample[2] - u)
            if not np.isfinite(prior) or not (self.log_mchirp_range[0] <= sample[0] <= self.log_mchirp_range[1]):
                continue
            log_w[n] = self.loglike(data, sample, u) + prior + np.log1p(q) + log_prior(sample) - log_g[n]
        log_z = logsumexp(log_w) - np.log(IS_SAMPLES)
        w = np.exp(log_w - log_w.max())
        return {"log_z_is": float(log_z), "ess": float(w.sum() ** 2 / (w**2).sum())}


def summary_of(lam: np.ndarray, truth: np.ndarray) -> dict:
    return {
        "mchirp": float(np.exp(lam[0])),
        "q": eta_to_q(lam[1]),
        "chi_eff": float(lam[2]),
        "d_log_mchirp": float(lam[0] - truth[0]),
        "d_eta": float(lam[1] - truth[1]),
    }


def pe_summary(outdir: Path, index: int) -> dict:
    post = bilby.result.read_in_result(str(outdir / f"inj{index}_aligned" / f"inj{index}_aligned_result.hdf5")).posterior
    q = post["mass_ratio"].to_numpy()
    chi_eff = (post["chi_1"].to_numpy() + q * post["chi_2"].to_numpy()) / (1 + q)
    best = post.loc[post["log_likelihood"].idxmax()]
    row = {}
    for name, values in (("mchirp", post["chirp_mass"].to_numpy()), ("q", q), ("chi_eff", chi_eff)):
        lo, mid, hi = np.percentile(values, (5, 50, 95))
        row.update({f"pe_{name}_lo": lo, f"pe_{name}_med": mid, f"pe_{name}_hi": hi})
    bq = best["mass_ratio"]
    row.update(
        pe_ml_mchirp=best["chirp_mass"],
        pe_ml_q=bq,
        pe_ml_chi_eff=(best["chi_1"] + bq * best["chi_2"]) / (1 + bq),
        pe_ml_chi_1=best["chi_1"],
        pe_ml_chi_2=best["chi_2"],
    )
    return row


@cache
def injection_snr(study: str, index: int) -> float:
    inj = STUDIES[study][index]
    if "target_snr" in inj:
        return inj["target_snr"]
    ifos = make_ifos(inj, inj["distance"])
    return float(np.sqrt(sum(ifo.meta_data["optimal_SNR"] ** 2 for ifo in ifos)))


@cache
def cached_model(study: str, index: int, approximant: str, optics: str) -> Model:
    return Model(STUDIES[study][index], approximant, injection_snr(study, index), optics)


def run_start(task: tuple) -> dict:
    """One local fit: align the extrinsics at the start point, then Levenberg-Marquardt."""
    study, index, optics, label, lmc, eta, chi = task
    model = cached_model(study, index, "xp", optics)
    data = model.lensed
    if lmc is None:
        lmc = model.best_log_mchirp(data, eta, chi)
    lam, loglike = model.fit(data, model.align(data, model.clip(np.array([lmc, eta, chi, 0, 0, 0]))))
    ff = model.fitting_factor(data, np.exp(lam[0]), lam[1], lam[2], lam[2])
    return {"index": index, "start": label, "loglike": loglike, "ff": ff, **summary_of(lam, model.truth)}


def compare(outdir: Path, study: str, indices: list[int], optics: str, workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, tasks = [], []
    for index in indices:
        inj = STUDIES[study][index]
        model = cached_model(study, index, "xp", optics)
        row = {"index": index, "m1": inj["m1"], "dt_ms": inj["dt"] * 1e3, "flux": inj["flux"],
               "snr": injection_snr(study, index)}  # fmt: skip
        row.update(pe_summary(outdir, index))
        for kind in LENSINGS:
            lin = summary_of(model.truth + model.linear_bias(model.data(inj["dt"], inj["flux"], kind)), model.truth)
            row.update({f"lin_{kind}_{k}": v for k, v in lin.items()})
        data = model.lensed
        row["ff_truth"] = model.fitting_factor(data, np.exp(model.truth[0]), model.truth[1], 0.0, 0.0)
        q = row["pe_ml_q"]
        row["ff_pe_ml"] = model.fitting_factor(
            data, row["pe_ml_mchirp"], q / (1 + q) ** 2, row["pe_ml_chi_1"], row["pe_ml_chi_2"]
        )
        rows.append(row)
        tasks.append((study, index, optics, "truth", *model.truth[:3]))
        tasks.append((study, index, optics, "pe_ml", np.log(row["pe_ml_mchirp"]), q / (1 + q) ** 2, row["pe_ml_chi_eff"]))
        tasks += [(study, index, optics, f"grid_q{sq}_chi{sc}", None, sq / (1 + sq) ** 2, sc)
                  for sq in START_Q for sc in START_CHI]  # fmt: skip
    with ProcessPoolExecutor(workers) as pool:
        fits = pd.DataFrame(list(pool.map(run_start, tasks, chunksize=1)))
    table = pd.DataFrame(rows)
    for index, group in fits.groupby("index"):
        best = group.loc[group["loglike"].idxmax()]
        ridge = group[group["loglike"] > best["loglike"] - RIDGE_DLNL]
        here = table["index"] == index
        for label, pick in (("best", best), ("from_truth", group[group["start"] == "truth"].iloc[0]),
                            ("from_ml", group[group["start"] == "pe_ml"].iloc[0])):  # fmt: skip
            for k in ("mchirp", "q", "chi_eff", "ff"):
                table.loc[here, f"{label}_{k}"] = pick[k]
        table.loc[here, "best_start"] = best["start"]
        for k in ("chi_eff", "q", "mchirp"):
            table.loc[here, f"ridge_{k}_min"] = ridge[k].min()
            table.loc[here, f"ridge_{k}_max"] = ridge[k].max()
        table.loc[here, "ridge_n"] = len(ridge)
    return table, fits


def profile_task(task: tuple) -> list[dict]:
    study, index, optics, kind, starts = task
    model = cached_model(study, index, "xp", optics)
    return [{"index": index, **row} for row in profile_cell(model, model.lensed, starts, kind, PROFILE_Q)]


def fixed_q_fit(model: Model, data: np.ndarray, q: float, start: np.ndarray, free: tuple) -> tuple[np.ndarray, float]:
    lam = start.copy()
    lam[1] = q / (1 + q) ** 2
    return model.fit(data, model.align(data, model.clip(lam)), free=free)


def same_mode(terms: dict, lam: np.ndarray, kept: tuple) -> bool:
    """Within two conditional sigmas of a kept solution in (log_mchirp, chi)."""
    other_terms, other = kept
    cov = np.array([[other_terms["var_log_mchirp"], other_terms["cov_log_mchirp_chi"]],
                    [other_terms["cov_log_mchirp_chi"], other_terms["var_chi"]]])  # fmt: skip
    delta = np.array([lam[0] - other[0], lam[2] - other[2]])
    return float(delta @ np.linalg.solve(cov, delta)) < SAME_MODE_D2


def profile_cell(model: Model, data: np.ndarray, starts: tuple, kind: str, q_grid: np.ndarray) -> list[dict]:
    """Fixed-q fits over the q grid with Laplace evidences (aligned or lensed template).

    Each q is first fitted from the global starts. Then, sweeping up and down the grid, any q whose best
    fit is worse than its neighbour's by more than CONTINUATION_GAP is refitted by following the neighbour's
    best solution in q steps of at most CONTINUATION_STEP, so a branch found at one q is tracked to the next.
    Solutions within two sigmas of a better one are merged.
    """
    free = LENSED_AT_FIXED_Q if kind == "lensed" else AT_FIXED_Q
    grid = [float(q) for q in q_grid]
    found = {}
    for q in grid:
        eta = q / (1 + q) ** 2
        if kind == "lensed":
            seeds = [np.array([model.truth[0], eta, 0.0, 0, 0, 0, *model.lens_truth])]
        else:
            seeds = [np.array([model.best_log_mchirp(data, eta, chi), eta, chi, 0, 0, 0]) for chi in starts]
        found[q] = [fixed_q_fit(model, data, q, seed, free) for seed in seeds]
    for _ in range(CONTINUATION_SWEEPS):
        changed = False
        for order in (grid, grid[::-1]):
            for prev, q in zip(order[:-1], order[1:]):
                seed, seed_loglike = max(found[prev], key=lambda x: x[1])
                here = max(loglike for _, loglike in found[q])
                if seed_loglike - here > CONTINUATION_GAP:
                    lam = seed
                    steps = max(2, int(np.ceil(abs(q - prev) / CONTINUATION_STEP)))
                    for sub in np.linspace(prev, q, steps + 1)[1:]:
                        lam, loglike = fixed_q_fit(model, data, float(sub), lam, free)
                    if loglike > here:
                        found[q].append((lam, loglike))
                        changed = True
        if not changed:
            break
    rows = []
    for q in grid:
        kept = []
        for lam, _ in sorted(found[q], key=lambda x: -x[1]):
            try:
                terms = model.laplace_terms(data, lam, free)
            except np.linalg.LinAlgError:
                continue
            if any(same_mode(terms, lam, k) for k in kept):
                continue
            kept.append((terms, lam))
        for terms, lam in kept:
            rows.append({
                "kind": kind, **terms, "q": q, "log_mchirp": lam[0], "tc": lam[3], "phic": lam[4],
                "log_amp": lam[5], "lens_dt": lam[6] if kind == "lensed" else np.nan,
                "lens_flux": lam[7] if kind == "lensed" else np.nan, "log_z": log_z_from(terms),
            })  # fmt: skip
    return rows


def q_weights(profile: pd.DataFrame, column: str = "log_z") -> pd.Series:
    grid = np.sort(profile["q"].unique())
    widths = dict(zip(grid, np.gradient(grid)))
    return profile[column] + log_prior_q(profile["q"]) + np.log(profile["q"].map(widths))


def laplace_posterior(profile: pd.DataFrame, column: str = "log_z", var_scale: float = 1.0) -> dict:
    """Mixture of conditional Gaussians over the q grid, weighted by evidence times the q prior."""
    rng = np.random.default_rng(1)
    profile = profile.assign(**{k: profile[k] * var_scale for k in ("var_log_mchirp", "var_chi", "cov_log_mchirp_chi")})
    weight = q_weights(profile, column)
    prob = np.exp(weight - weight.max())
    prob /= prob.sum()
    rows = profile.iloc[rng.choice(len(profile), POSTERIOR_SAMPLES, p=prob)]
    mean = rows[["log_mchirp", "chi_eff"]].to_numpy()
    cov = np.stack([rows[["var_log_mchirp", "cov_log_mchirp_chi"]].to_numpy(),
                    rows[["cov_log_mchirp_chi", "var_chi"]].to_numpy()], axis=1)  # fmt: skip
    draws = mean + np.einsum("nij,nj->ni", np.linalg.cholesky(cov), rng.standard_normal((len(rows), 2)))
    samples = {"q": rows["q"].to_numpy(), "mchirp": np.exp(draws[:, 0]), "chi_eff": np.clip(draws[:, 1], -1, 1)}
    out = {"log_z": float(logsumexp(weight))}
    for name, values in samples.items():
        lo, mid, hi = np.percentile(values, (5, 50, 95))
        out.update({f"{name}_lo": lo, f"{name}_med": mid, f"{name}_hi": hi})
    low = np.ones(POSTERIOR_SAMPLES, bool)
    for name, edge in LOW_Q_MODE.items():
        low &= samples[name] < edge
    out["frac_low_q_mode"] = float(low.mean())
    return out


def profile(table: pd.DataFrame, fits: pd.DataFrame, study: str, optics: str,
            workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:  # fmt: skip
    """Laplace posterior and evidence of the aligned model, and the evidence of the lensed model."""
    tasks = []
    for index, group in fits.groupby("index"):
        modes = group.sort_values("loglike", ascending=False).round({"chi_eff": 2}).drop_duplicates("chi_eff")
        starts = tuple(modes["chi_eff"].head(PROFILE_STARTS))
        tasks += [(study, index, optics, kind, starts) for kind in ("aligned", "lensed")]
    with ProcessPoolExecutor(workers) as pool:
        solutions = pd.DataFrame([row for rows in pool.map(profile_task, tasks, chunksize=1) for row in rows])
    summary = []
    for index, group in solutions.groupby("index"):
        aligned = laplace_posterior(group[group["kind"] == "aligned"])
        lensed = laplace_posterior(group[group["kind"] == "lensed"])
        summary.append({"index": index, **{f"lap_{k}": v for k, v in aligned.items()},
                        "lap_lnB_lensed_vs_aligned": lensed["log_z"] - aligned["log_z"]})  # fmt: skip
    return table.merge(pd.DataFrame(summary), on="index"), solutions


def importance_task(task: tuple) -> dict:
    study, index, optics, row = task
    model = cached_model(study, index, "xp", optics)
    q = row["q"]
    lam = np.array([row["log_mchirp"], q / (1 + q) ** 2, row["chi_eff"], row["tc"], row["phic"], row["log_amp"]])
    rng = np.random.default_rng(int(1e6 * q) + index)
    return {**row, **model.importance(model.lensed, lam, rng)}


def importance_check(solutions: pd.DataFrame, study: str, indices: list[int], optics: str, workers: int) -> tuple:
    """Re-weight the aligned Laplace mixture with importance-sampled conditional evidences."""
    picked = solutions[(solutions["kind"] == "aligned") & solutions["index"].isin(indices)]
    tasks = [(study, int(r["index"]), optics, r.to_dict()) for _, r in picked.iterrows()]
    with ProcessPoolExecutor(workers) as pool:
        checked = pd.DataFrame(list(pool.map(importance_task, tasks, chunksize=1)))
    summary = []
    for index, group in checked.groupby("index"):
        summary.append({"index": index, **{f"is_{k}": v for k, v in laplace_posterior(group, "log_z_is").items()},
                        "min_ess": group["ess"].min(), "max_dlogz": (group["log_z_is"] - group["log_z"]).abs().max()})  # fmt: skip
    return pd.DataFrame(summary), checked


def scan(study: str) -> pd.DataFrame:
    """Linear bias per unit sqrt(I) against the lens delay, one source per mass."""
    rows = []
    seen = set()
    for inj in STUDIES[study]:
        if inj["m1"] in seen:
            continue
        seen.add(inj["m1"])
        for approximant in ("xp", "taylorf2") if inj["m1"] < 50 else ("xp",):
            model = Model(inj, approximant)
            _, d = model.derivatives(model.truth)
            gamma = model.fisher(d)
            for dt in SCAN_DT:
                for kind in LENSINGS:
                    # F - 1 is linear in sqrt(I); the split pieces are not, so take their small-I slope.
                    delta = model.h0 * (lensing(model.f, dt, SCAN_FLUX, kind) - 1.0) / np.sqrt(SCAN_FLUX)
                    bias = np.linalg.solve(gamma, model.inner(d, delta))
                    rows.append({"m1": inj["m1"], "approximant": approximant, "dt_ms": dt * 1e3, "kind": kind,
                                 **{f"d_{n}": b for n, b in zip(NAMES, bias)}})  # fmt: skip
    return pd.DataFrame(rows)


def toy() -> tuple[pd.DataFrame, pd.DataFrame]:
    """First-order bias of a light q < 1 binary, TaylorF2 toy vs XP, in full and reduced parameter sets."""
    rows, info = [], []
    for approximant in ("taylorf2", "xp"):
        model = Model({**TOY, "dt": 0.0, "flux": 0.0}, approximant)
        _, d = model.derivatives(model.truth)
        gamma = model.fisher(d)
        for name, keep in TOY_SETS.items():
            sub = np.ix_(keep, keep)
            sigma = np.sqrt(np.diag(np.linalg.inv(gamma[sub])))
            info.append({"approximant": approximant, "set": name, "cond": np.linalg.cond(gamma[sub]),
                         **{f"sigma_{NAMES[k]}": v for k, v in zip(keep, sigma) if k < 3}})  # fmt: skip
        for dt in SCAN_DT:
            for kind in LENSINGS:
                delta = model.h0 * (lensing(model.f, dt, SCAN_FLUX, kind) - 1.0) / np.sqrt(SCAN_FLUX)
                proj = model.inner(d, delta)
                for name, keep in TOY_SETS.items():
                    bias = np.linalg.solve(gamma[np.ix_(keep, keep)], proj[list(keep)])
                    out = dict(zip((NAMES[k] for k in keep), bias))
                    rows.append({"approximant": approximant, "set": name, "dt_ms": dt * 1e3, "kind": kind,
                                 **{f"d_{n}": out.get(n, 0.0) for n in NAMES[:3]}})  # fmt: skip
    return pd.DataFrame(rows), pd.DataFrame(info)


def toy_report(curves: pd.DataFrame, info: pd.DataFrame) -> None:
    print(info.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    curves = curves.assign(dt_ms=curves["dt_ms"].round(1))
    for name in TOY_SETS:
        sub = curves[curves["set"] == name]
        for col in ("d_log_mchirp", "d_eta", "d_chi"):
            if name == "fix_chi" and col == "d_chi" or name == "fix_eta" and col == "d_eta":
                continue
            tf2 = sub[(sub["approximant"] == "taylorf2") & (sub["kind"] == "full")].set_index("dt_ms")[col]
            xp = sub[(sub["approximant"] == "xp") & (sub["kind"] == "full")].set_index("dt_ms")[col]
            amp = sub[(sub["approximant"] == "taylorf2") & (sub["kind"] == "amplitude")][col].abs().max()
            big = xp.abs() > 0.1 * xp.abs().max()
            agree = float(np.mean(np.sign(tf2[big]) == np.sign(xp[big])))
            corr = float(np.corrcoef(tf2, xp)[0, 1])
            print(f"{name:8s} {col:13s} TaylorF2 vs XP: corr {corr:+.3f}, sign agreement {agree:.2f} "
                  f"(|XP| > 10% max); TaylorF2 amplitude-only max/full max {amp / tf2.abs().max():.1e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("results/pe_generality"))
    parser.add_argument("--study", choices=sorted(STUDIES), default="generality")
    parser.add_argument("--indices", type=int, nargs="*")
    parser.add_argument("--optics", choices=("geometric", "wave"), default="geometric")
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--importance", type=int, nargs="*", default=[])
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--toy", action="store_true")
    args = parser.parse_args()
    tag = f"{args.study}_{args.optics}"
    indices = args.indices if args.indices else list(range(len(STUDIES[args.study])))
    if args.toy:
        curves, info = toy()
        curves.to_csv(args.outdir / "predictor_toy.csv", index=False)
        toy_report(curves, info)
    table, fits = compare(args.outdir, args.study, indices, args.optics, args.workers)
    fits.to_csv(args.outdir / f"predictor_fits_{tag}.csv", index=False)
    table, solutions = profile(table, fits, args.study, args.optics, args.workers)
    solutions.to_csv(args.outdir / f"predictor_profile_{tag}.csv", index=False)
    if args.importance:
        checked_summary, checked = importance_check(solutions, args.study, args.importance, args.optics, args.workers)
        checked.to_csv(args.outdir / f"predictor_importance_{tag}.csv", index=False)
        table = table.merge(checked_summary, on="index", how="left")
    table.to_csv(args.outdir / f"predictor_{tag}.csv", index=False)
    if args.scan:
        scan(args.study).to_csv(args.outdir / "predictor_scan.csv", index=False)
    cols = ["index", "m1", "dt_ms", "flux", "snr", "pe_chi_eff_med", "best_chi_eff", "pe_q_med", "best_q",
            "pe_mchirp_med", "best_mchirp", "ff_truth", "ff_pe_ml", "best_ff", "lap_lnB_lensed_vs_aligned"]  # fmt: skip
    print(table[cols].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    cols = ["index"] + [f"{src}_{k}_{p}" for k in ("chi_eff", "q", "mchirp") for src in ("pe", "lap")
                        for p in ("lo", "med", "hi")] + ["lap_frac_low_q_mode"]  # fmt: skip
    print(table[cols].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    if args.importance:
        cols = ["index", "pe_chi_eff_med", "lap_chi_eff_med", "is_chi_eff_med", "pe_q_med", "lap_q_med", "is_q_med",
                "lap_frac_low_q_mode", "is_frac_low_q_mode", "min_ess", "max_dlogz"]  # fmt: skip
        print(table[table["index"].isin(args.importance)][cols].to_string(index=False, float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    main()
