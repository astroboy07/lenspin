import lal
import lalsimulation as ls
import numpy as np

F_REF = 20.0
D_REF_MPC = 1000.0
APPROXIMANT = ls.IMRPhenomXP


def chirp_mass(m1: float, m2: float) -> float:
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def component_masses(mchirp: float, q: float) -> tuple[float, float]:
    m1 = mchirp * (1.0 + q) ** 0.2 / q**0.6
    return m1, q * m1


def fd_polarizations(approximant, m1, m2, spin1, spin2, incl, grid):
    hp, hc = ls.SimInspiralChooseFDWaveform(
        m1 * lal.MSUN_SI,
        m2 * lal.MSUN_SI,
        *spin1,
        *spin2,
        D_REF_MPC * 1e6 * lal.PC_SI,
        incl,
        0.0,
        0.0,
        0.0,
        0.0,
        grid.df,
        grid.f_min,
        grid.f_max,
        F_REF,
        None,
        approximant,
    )
    band = slice(grid.k_min, grid.k_max + 1)
    return hp.data.data[band].copy(), hc.data.data[band].copy()


def aligned_polarizations(m1, m2, chi1z, chi2z, incl, grid):
    return fd_polarizations(APPROXIMANT, m1, m2, (0.0, 0.0, chi1z), (0.0, 0.0, chi2z), incl, grid)


def precessing_polarizations(x, grid):
    mchirp, q, chi1, cos_tilt1, chi2z, theta_jn, phi_jl = x
    m1, m2 = component_masses(mchirp, q)
    incl, *spins = ls.SimInspiralTransformPrecessingNewInitialConditions(
        theta_jn,
        phi_jl,
        np.arccos(np.clip(cos_tilt1, -1.0, 1.0)),
        0.0 if chi2z >= 0 else np.pi,
        0.0,
        chi1,
        abs(chi2z),
        m1 * lal.MSUN_SI,
        m2 * lal.MSUN_SI,
        F_REF,
        0.0,
    )
    return fd_polarizations(APPROXIMANT, m1, m2, spins[:3], spins[3:], incl, grid)


def lens_factor(freqs, dt, flux_ratio):
    # LAL transforms with e^{-2 pi i f t}, so the delayed saddle (type II) image carries +i.
    return 1.0 + 1j * np.sqrt(flux_ratio) * np.exp(-2j * np.pi * freqs * dt)
