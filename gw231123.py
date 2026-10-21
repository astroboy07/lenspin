import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from maps import append, run_cell
from rates import point_lens_observables

# NRSur7dq4 isolated point-lens fit of GW231123 (Goyal et al., arXiv:2512.17631, Tables 1 and S1).
LENS_MASS, LENS_Y = 821.0, 0.68
TOTAL_MASS = 300.0
TRUE_Q = (0.5, 0.75, 1.0)
FLUXES = (0.26, 0.30)
OPTICS = ("geometric", "wave")
TRUE_CHI_EFF = (0.0, 0.36)
SNR = 20.0
# Published detector-frame chirp mass, q and chi_eff, unlensed vs isolated point lens (NRSur7dq4).
PUBLISHED = {"unlensed": (139.0, 0.83, 0.30), "lensed": (123.0, 0.53, 0.36)}


def cells() -> list[dict]:
    dt, _ = point_lens_observables(LENS_MASS, LENS_Y)
    grid = itertools.product(TRUE_Q, FLUXES, OPTICS, TRUE_CHI_EFF)
    return [{"total_mass": TOTAL_MASS, "q_true": q, "dt_ms": round(dt, 2), "flux": f, "optics": o, "chi_eff_true": c}
            for q, f, o, c in grid]  # fmt: skip


def show(table: pd.DataFrame) -> None:
    rows = table[table["snr"] == SNR].sort_values(["chi_eff_true", "q_true", "flux", "optics"])
    for r in rows.to_dict("records"):
        print(f"q {r['q_true']:.2f} I {r['flux']:.2f} {r['optics']:9s} chi_true {r['chi_eff_true']:.2f} | "
              f"Mc {r['mchirp_med']:.1f} [{r['mchirp_lo']:.1f},{r['mchirp_hi']:.1f}] vs {r['mchirp_true']:.1f} "
              f"({r['mchirp_med'] / r['mchirp_true'] - 1:+.0%}) | q {r['q_med']:.2f} [{r['q_lo']:.2f},{r['q_hi']:.2f}] | "
              f"chi_eff {r['chi_eff_med']:+.2f} [{r['chi_eff_lo']:+.2f},{r['chi_eff_hi']:+.2f}] | "
              f"lnB {r['lnB_lensed_vs_aligned']:.1f} perfect {r['lnB_perfect_lens']:.1f}")  # fmt: skip
    (mc_u, q_u, chi_u), (mc_l, q_l, chi_l) = PUBLISHED["unlensed"], PUBLISHED["lensed"]
    print(f"published (NRSur7dq4): Mc {mc_u:g} vs {mc_l:g} ({mc_u / mc_l - 1:+.0%}), q {q_u} vs {q_l}, "
          f"chi_eff {chi_u} vs {chi_l}")  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/gw231123"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary, solutions = args.out / "cells.csv", args.out / "solutions.csv"
    if not summary.exists():
        with ProcessPoolExecutor(args.workers) as pool:
            for rows, sols in pool.map(run_cell, cells()):
                append(summary, rows)
                append(solutions, sols)
    show(pd.read_csv(summary))


if __name__ == "__main__":
    main()
