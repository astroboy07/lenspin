import mpmath
import numpy as np

NODES = 8192
DPS = 20


def magnifications(y: float) -> tuple[float, float]:
    """Point-lens image magnifications (mu_plus, |mu_minus|)."""
    s = (y**2 + 2.0) / (y * np.sqrt(y**2 + 4.0))
    return 0.5 * (s + 1.0), 0.5 * (s - 1.0)


def scaled_delay(y: float) -> float:
    """Time delay between the two images in units of 4 G M_Lz / c^3."""
    r = np.sqrt(y**2 + 4.0)
    return 0.5 * y * r + np.log((r + y) / (r - y))


def impact_from_flux(flux: float) -> float:
    """Impact parameter y giving the flux ratio |mu_minus| / mu_plus."""
    s = (1.0 + flux) / (1.0 - flux)
    return float(np.sqrt(2.0 * (s / np.sqrt(s**2 - 1.0) - 1.0)))


def lens_parameters(dt: float, flux: float) -> tuple[float, float]:
    """Redshifted lens mass in seconds (G M_Lz / c^3) and y that reproduce the geometric (dt, flux)."""
    y = impact_from_flux(flux)
    return dt / (4.0 * scaled_delay(y)), y


def amplification(w: float, y: float) -> complex:
    """Takahashi & Nakamura (2003) point-lens amplification F(w, y), e^{+2 pi i f t} convention."""
    xm = 0.5 * (y + np.sqrt(y**2 + 4.0))
    phim = 0.5 * (xm - y) ** 2 - np.log(xm)
    with mpmath.workdps(DPS):
        half = mpmath.mpf(w) / 2
        value = (
            mpmath.exp(mpmath.pi * w / 4 + 1j * half * (mpmath.log(half) - 2 * phim))
            * mpmath.gamma(1 - 1j * half)
            * mpmath.hyp1f1(1j * half, 1, 1j * half * y**2)
        )
    return complex(value)


def wave_lens_factor(freqs: np.ndarray, dt: float, flux: float) -> np.ndarray:
    """Wave-optics factor in the LAL e^{-2 pi i f t} convention, normalised so its geometric limit is
    1 + i sqrt(flux) e^{-2 pi i f dt} (the same as signals.lens_factor)."""
    mass, y = lens_parameters(dt, flux)
    nodes = np.linspace(freqs[0], freqs[-1], NODES)
    values = np.array([np.conj(amplification(8.0 * np.pi * mass * f, y)) for f in nodes])
    values /= np.sqrt(magnifications(y)[0])
    return np.interp(freqs, nodes, values.real) + 1j * np.interp(freqs, nodes, values.imag)
