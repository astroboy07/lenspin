import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.cosmology import Planck18

from lenspin.bias.maps import SNRS
from lenspin.bias.maps_summary import MTSUN_MS, flags
from lenspin.models.waveoptics import lens_parameters, magnifications, scaled_delay

MSUN_S = MTSUN_MS * 1e-3
C_KMS = 299792.458
DANGER_SNRS = (8.0, 10.0, 12.0)
# GWTC-4.0 isolated point-mass fits of GW231123 (arXiv:2512.16347): redshifted lens mass [Msun] and y.
GW231123_FITS = {"IMRPhenomXPHM-ST": (911.0, 0.61), "NRSur7dq4": (803.0, 0.69)}
# Detected SNRs follow p(rho) ~ rho^-4 above 8; each map SNR stands for the bin around it.
SNR_EDGES = (8.0, 9.0, 11.0, 13.5, 17.5, 22.5, 30.0, 42.0, np.inf)
LEVELS = ("biased", "danger_8", "danger_3")
LENS_MASSES = np.geomspace(10.0, 1e4, 31)
SOURCE_Z = (0.5, 1.0, 2.0)
Y_GRID = np.linspace(1e-3, 3.0, 1500)
LOG_MLZ = np.linspace(np.log(1.0), np.log(1e5), 600)
Z_STEPS = 400


def danger_lenses(cells: pd.DataFrame, level: float = 8.0) -> pd.DataFrame:
    """Redshifted lens mass and impact parameter of every biased-and-hidden cell."""
    danger = flags(cells)
    danger = danger[danger[f"danger_{level:g}"] & danger["snr"].isin(DANGER_SNRS)].copy()
    lens = [lens_parameters(dt * 1e-3, flux) for dt, flux in zip(danger["dt_ms"], danger["flux"])]
    danger["lens_mass"] = [m / MSUN_S for m, _ in lens]
    danger["y"] = [y for _, y in lens]
    return danger


def point_lens_observables(lens_mass: float, y: float) -> tuple[float, float]:
    """Geometric delay [ms] and flux ratio of a point lens of redshifted mass lens_mass [Msun]."""
    mu_plus, mu_minus = magnifications(y)
    return 4.0 * lens_mass * MSUN_S * scaled_delay(y) * 1e3, mu_minus / mu_plus


def snr_weights() -> np.ndarray:
    lo, hi = np.array(SNR_EDGES[:-1]), np.array(SNR_EDGES[1:])
    return (SNR_EDGES[0] / lo) ** 3 - (SNR_EDGES[0] / hi) ** 3


def lens_kernel(z_source: float) -> tuple[np.ndarray, np.ndarray]:
    """d tau / d z_lens for compact objects making up all of the dark matter, per unit y^2 of cross-section."""
    z = np.linspace(0.0, z_source, Z_STEPS)
    dc, dcs = Planck18.comoving_distance(z).value, Planck18.comoving_distance(z_source).value
    hubble = Planck18.H(z).value
    kernel = 1.5 * Planck18.Odm0 * (Planck18.H0.value / C_KMS) ** 2 * (C_KMS / hubble) * (1 + z) * dc * (dcs - dc) / dcs
    return z, kernel


def edges(values: np.ndarray, log: bool = False) -> np.ndarray:
    v = np.log(values) if log else np.asarray(values, float)
    mid = (v[1:] + v[:-1]) / 2
    e = np.concatenate([[2 * v[0] - mid[0]], mid, [2 * v[-1] - mid[-1]]])
    return np.exp(e) if log else e


def flagged_area(grid: pd.DataFrame) -> np.ndarray:
    """Cross-section in y^2 of the flagged (delay, flux) cells, on the redshifted-lens-mass grid LOG_MLZ."""
    dt_edges, flux_edges = edges(grid.columns.to_numpy()), edges(grid.index.to_numpy(), log=True)
    flagged = grid.to_numpy(bool)
    mu_plus, mu_minus = magnifications(Y_GRID)
    i = np.searchsorted(flux_edges, mu_minus / mu_plus) - 1
    delays = 4.0 * np.exp(LOG_MLZ)[:, None] * MSUN_S * scaled_delay(Y_GRID)[None, :] * 1e3
    j = np.searchsorted(dt_edges, delays) - 1
    inside = (i >= 0) & (i < flagged.shape[0]) & (j >= 0) & (j < flagged.shape[1])
    hit = np.where(inside, flagged[np.clip(i, 0, flagged.shape[0] - 1), np.clip(j, 0, flagged.shape[1] - 1)], False)
    return np.trapezoid(2.0 * Y_GRID * hit, Y_GRID, axis=1)


def depths(cells: pd.DataFrame) -> pd.DataFrame:
    """Optical depth (per unit compact-object fraction f) of the flagged cells for monochromatic lenses."""
    kernels = {zs: lens_kernel(zs) for zs in SOURCE_Z}
    rows = []
    for (mass, q, snr), cell in flags(cells).groupby(["total_mass", "q_true", "snr"]):
        for level in LEVELS:
            area = flagged_area(cell.pivot(index="flux", columns="dt_ms", values=level))
            for zs, lens_mass in itertools.product(SOURCE_Z, LENS_MASSES):
                z, kernel = kernels[zs]
                a = np.interp(np.log(lens_mass * (1 + z)), LOG_MLZ, area)
                rows.append({"total_mass": mass, "q_true": q, "snr": snr, "level": level, "z_source": zs,
                             "lens_mass": lens_mass, "depth": np.trapezoid(kernel * a, z)})  # fmt: skip
    return pd.DataFrame(rows)


def per_event(table: pd.DataFrame) -> pd.DataFrame:
    """Probability per detected event and unit f, averaging the SNR over p(rho) ~ rho^-4."""
    weights = dict(zip(SNRS, snr_weights()))
    table = table.assign(weighted=table["depth"] * table["snr"].map(weights))
    return table.groupby(["total_mass", "q_true", "level", "z_source", "lens_mass"], as_index=False)["weighted"].sum()


def show(danger: pd.DataFrame) -> None:
    group = danger.groupby(["snr", "total_mass", "q_true"])
    table = group.agg(cells=("dt_ms", "size"), dt_lo=("dt_ms", "min"), dt_hi=("dt_ms", "max"),
                      dt_m_lo=("dt_over_m", "min"), dt_m_hi=("dt_over_m", "max"), flux_lo=("flux", "min"),
                      flux_hi=("flux", "max"), mass_lo=("lens_mass", "min"), mass_med=("lens_mass", "median"),
                      mass_hi=("lens_mass", "max"), y_lo=("y", "min"), y_hi=("y", "max"))  # fmt: skip
    print(table.round(2).to_string())
    for name, (mass, y) in GW231123_FITS.items():
        dt, flux = point_lens_observables(mass, y)
        print(f"GW231123 {name}: M_Lz {mass:g} Msun, y {y}: dt {dt:.1f} ms, I {flux:.2f}")


def show_depths(events: pd.DataFrame) -> None:
    for zs in SOURCE_Z:
        z, kernel = lens_kernel(zs)
        print(f"tau(y < 1, f = 1, z_s = {zs:g}) = {np.trapezoid(kernel, z):.2e}")
    best = events[events["z_source"] == 1.0].sort_values("weighted").groupby(["total_mass", "q_true", "level"]).tail(1)
    print(best.sort_values(["level", "total_mass", "q_true"]).to_string(index=False, float_format="%.3g"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=Path, default=Path("results/maps/cells.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/rates"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(args.cells)
    cells = cells[cells["optics"] == "geometric"]
    for level in (8.0, 3.0):
        danger = danger_lenses(cells, level)
        danger.to_csv(args.out / f"danger_lenses_{level:g}.csv", index=False)
        print(f"===== biased and lnB_perfect < {level:g}")
        show(danger)
    table = depths(cells)
    table.to_csv(args.out / "depths.csv", index=False)
    events = per_event(table)
    events.to_csv(args.out / "per_event.csv", index=False)
    print("===== per-event probability per unit f (z_s = 1, best lens mass)")
    show_depths(events)


if __name__ == "__main__":
    main()
