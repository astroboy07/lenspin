from dataclasses import dataclass

import lal
import lalsimulation as ls
import numpy as np
from scipy.optimize import minimize_scalar

NETWORKS = {
    "H": ("H1",),
    "HL": ("H1", "L1"),
    "HLV": ("H1", "L1", "V1"),
    "HLVI": ("H1", "L1", "V1", "I1"),
    "ET": ("E1", "E2", "E3"),
}
PSD_MODELS = {
    "H1": ls.SimNoisePSDaLIGOAPlusDesignSensitivityT1800042,
    "L1": ls.SimNoisePSDaLIGOAPlusDesignSensitivityT1800042,
    "I1": ls.SimNoisePSDaLIGOAPlusDesignSensitivityT1800042,
    "V1": ls.SimNoisePSDAdVDesignSensitivityP1200087,
    "E1": ls.SimNoisePSDEinsteinTelescopeP1600143,
    "E2": ls.SimNoisePSDEinsteinTelescopeP1600143,
    "E3": ls.SimNoisePSDEinsteinTelescopeP1600143,
}
F_MIN = {"ET": 10.0}
F_MIN_DEFAULT = 20.0
F_MAX = 1024.0
GPS_TIME = 1_400_000_000
UPSAMPLE = 8
DEGENERATE_TOL = 1e-10
DETECTORS = {d.frDetector.prefix: d for d in lal.CachedDetectors}


@dataclass(frozen=True)
class FrequencyGrid:
    df: float
    k_min: int
    k_max: int

    @property
    def f_min(self) -> float:
        return self.k_min * self.df

    @property
    def f_max(self) -> float:
        return self.k_max * self.df

    @property
    def freqs(self) -> np.ndarray:
        return np.arange(self.k_min, self.k_max + 1) * self.df


def make_grid(f_min: float, f_max: float, m1_min: float, m2_min: float) -> FrequencyGrid:
    tchirp = ls.SimInspiralChirpTimeBound(f_min, m1_min * lal.MSUN_SI, m2_min * lal.MSUN_SI, 0.0, 0.0)
    duration = 2.0 ** np.ceil(np.log2(1.5 * tchirp + 2.0))
    df = 1.0 / duration
    return FrequencyGrid(df, round(f_min / df), round(f_max / df))


def noise_weights(model, grid: FrequencyGrid) -> np.ndarray:
    series = lal.CreateREAL8FrequencySeries(
        "psd", lal.LIGOTimeGPS(0), 0.0, grid.df, lal.DimensionlessUnit, grid.k_max + 2
    )
    model(series, grid.f_min)
    psd = series.data.data[grid.k_min : grid.k_max + 1]
    return np.where(psd > 0, 4.0 * grid.df / np.where(psd > 0, psd, 1.0), 0.0)


class Network:
    def __init__(self, name: str, grid: FrequencyGrid) -> None:
        self.name = name
        self.grid = grid
        self.freqs = grid.freqs
        self.dets = [DETECTORS[p] for p in NETWORKS[name]]
        self.weights = [noise_weights(PSD_MODELS[p], grid) for p in NETWORKS[name]]
        self.gps = lal.LIGOTimeGPS(GPS_TIME)
        self.gmst = lal.GreenwichMeanSiderealTime(self.gps)
        self.nfft = UPSAMPLE * 2 ** int(np.ceil(np.log2(grid.k_max + 1)))
        self.step = 1.0 / (self.nfft * grid.df)

    def geometry(self, ra: float, dec: float) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        fp, fc, shifts = [], [], []
        for det in self.dets:
            f_plus, f_cross = lal.ComputeDetAMResponse(det.response, ra, dec, 0.0, self.gmst)
            tau = lal.TimeDelayFromEarthCenter(det.location, ra, dec, self.gps)
            fp.append(f_plus)
            fc.append(f_cross)
            shifts.append(np.exp(-2j * np.pi * self.freqs * tau))
        return np.array(fp), np.array(fc), shifts

    def project(self, hp, hc, ra, dec, psi) -> list[np.ndarray]:
        fp0, fc0, shifts = self.geometry(ra, dec)
        c, s = np.cos(2 * psi), np.sin(2 * psi)
        fp, fc = fp0 * c + fc0 * s, -fp0 * s + fc0 * c
        return [(fp[i] * hp + fc[i] * hc) * shifts[i] for i in range(len(self.dets))]

    def norm2(self, hs) -> float:
        return float(sum(np.sum(w * np.abs(h) ** 2) for w, h in zip(self.weights, hs)))

    def snr_cumulative(self, hs) -> np.ndarray:
        return np.cumsum(sum(w * np.abs(h) ** 2 for w, h in zip(self.weights, hs)))

    def series(self, x: np.ndarray) -> np.ndarray:
        full = np.zeros(self.nfft, dtype=complex)
        full[self.grid.k_min : self.grid.k_max + 1] = x
        return self.nfft * np.fft.ifft(full)

    def direct(self, x: np.ndarray, t: float) -> np.ndarray:
        return np.atleast_1d(np.sum(x * np.exp(2j * np.pi * self.freqs * t)))

    def maximize_over_time(self, xs, statistic) -> float:
        values = statistic(*[self.series(x) for x in xs])
        j = int(np.argmax(values))
        t0 = j * self.step
        res = minimize_scalar(
            lambda t: -statistic(*[self.direct(x, t) for x in xs])[0],
            bounds=(t0 - self.step, t0 + self.step),
            method="bounded",
            options={"xatol": self.step * 1e-3},
        )
        return max(float(values[j]), -float(res.fun))


def aligned_fstat(net: Network, signal, h0, ra, dec) -> float:
    fp, fc, shifts = net.geometry(ra, dec)
    xd = [w * s * np.conj(h0 * sh) for w, s, sh in zip(net.weights, signal, shifts)]
    sig2 = np.array([np.sum(w * np.abs(h0) ** 2) for w in net.weights])
    m = np.array(
        [
            [np.sum(fp * fp * sig2), np.sum(fp * fc * sig2)],
            [np.sum(fp * fc * sig2), np.sum(fc * fc * sig2)],
        ],
    )
    minv = np.linalg.pinv(m, rcond=1e-10, hermitian=True)
    xp = sum(f * x for f, x in zip(fp, xd))
    xc = sum(f * x for f, x in zip(fc, xd))

    def statistic(zp, zc):
        cross = np.real(np.conj(zp) * zc)
        return minv[0, 0] * np.abs(zp) ** 2 + minv[1, 1] * np.abs(zc) ** 2 + 2 * minv[0, 1] * cross

    return net.maximize_over_time([xp, xc], statistic)


def precessing_overlap(net: Network, signal, hp, hc, ra, dec) -> float:
    fp, fc, shifts = net.geometry(ra, dec)
    us = [(fp[i] * hp + fc[i] * hc) * shifts[i] for i in range(len(shifts))]
    vs = [(fc[i] * hp - fp[i] * hc) * shifts[i] for i in range(len(shifts))]
    quu, qvv = net.norm2(us), net.norm2(vs)
    quv = float(sum(np.sum(w * np.real(u * np.conj(v))) for w, u, v in zip(net.weights, us, vs)))
    det_q = quu * qvv - quv**2
    xu = sum(w * s * np.conj(u) for w, s, u in zip(net.weights, signal, us))
    xv = sum(w * s * np.conj(v) for w, s, v in zip(net.weights, signal, vs))
    if det_q <= DEGENERATE_TOL * quu * qvv:
        return net.maximize_over_time([xu], lambda zu: np.abs(zu) ** 2 / quu)

    def statistic(zu, zv):
        p11, p22 = np.abs(zu) ** 2, np.abs(zv) ** 2
        p12 = np.real(zu * np.conj(zv))
        b = p11 * qvv + p22 * quu - 2 * p12 * quv
        c = p11 * p22 - p12**2
        return (b + np.sqrt(np.maximum(b**2 - 4 * det_q * c, 0.0))) / (2 * det_q)

    return net.maximize_over_time([xu, xv], statistic)
