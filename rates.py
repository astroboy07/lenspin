import argparse
from pathlib import Path

import pandas as pd

from maps_summary import MTSUN_MS, flags
from waveoptics import lens_parameters, magnifications, scaled_delay

MSUN_S = MTSUN_MS * 1e-3
DANGER_SNRS = (8.0, 10.0, 12.0)
# GWTC-4.0 isolated point-mass fits of GW231123 (arXiv:2512.16347): redshifted lens mass [Msun] and y.
GW231123_FITS = {"IMRPhenomXPHM-ST": (911.0, 0.61), "NRSur7dq4": (803.0, 0.69)}


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


if __name__ == "__main__":
    main()
