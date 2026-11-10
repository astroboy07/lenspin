import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from lenspin.bias.predictor import Model, laplace_posterior, log_evidence, log_z_from, profile_cell, summary_of

REFERENCE_SNR = 25.0
SNRS = (8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 35.0, 50.0)
DT_MS = np.arange(0.5, 40.0, 1.0)
FLUXES = (0.05, 0.1, 0.2, 0.35, 0.6)
DENSE_FLUXES = tuple(float(f) for f in np.round(np.geomspace(0.03, 0.9, 20), 4))
TOTAL_MASSES = (30.0, 60.0, 100.0, 180.0)
TRUE_Q = (1.0, 0.5)
OPTICS = ("geometric", "wave")
MCHIRP_SPAN = (0.4, 1.6)
MAP_Q = np.unique(np.concatenate([np.geomspace(0.05, 0.5, 9), np.linspace(0.5, 1.0, 11)]))
START_Q = (0.15, 0.35, 0.6, 1.0)
START_CHI = (-0.5, 0.0, 0.5)
MAX_STARTS = 4
KEYS = ("total_mass", "q_true", "dt_ms", "flux", "optics")


def cells(masses: tuple = TOTAL_MASSES, dt_step: float = 1.0, fluxes: tuple = FLUXES, optics: tuple = OPTICS) -> list[dict]:
    """Lightest systems first, so the slowest cells start early."""
    delays = DT_MS[:: max(1, round(dt_step))]
    grid = itertools.product(sorted(masses), TRUE_Q, optics, fluxes, delays)
    return [dict(zip(("total_mass", "q_true", "optics", "flux", "dt_ms"), (m, q, o, f, round(float(t), 3))))
            for m, q, o, f, t in grid]  # fmt: skip


def injection(cell: dict) -> dict:
    m1 = cell["total_mass"] / (1.0 + cell["q_true"])
    return {"m1": m1, "m2": cell["q_true"] * m1, "dt": cell["dt_ms"] * 1e-3, "flux": cell["flux"],
            "mchirp_span": MCHIRP_SPAN, "target_snr": REFERENCE_SNR, "chi_eff": cell.get("chi_eff_true", 0.0)}  # fmt: skip


def run_cell(cell: dict) -> tuple[list[dict], list[dict]]:
    """Multi-start aligned fit, q-profiled Laplace evidences of the aligned and lensed models, then
    the aligned posterior and ln B(lensed vs aligned) at every SNR in SNRS."""
    model = Model(injection(cell), "xp", REFERENCE_SNR, cell["optics"])
    data = model.lensed
    starts = [tuple(model.truth[:3])]
    starts += [(None, q / (1 + q) ** 2, chi) for q in START_Q for chi in START_CHI]
    fits = []
    for lmc, eta, chi in starts:
        lmc = model.best_log_mchirp(data, eta, chi) if lmc is None else lmc
        lam, loglike = model.fit(data, model.align(data, model.clip(np.array([lmc, eta, chi, 0, 0, 0]))))
        fits.append((loglike, lam))
    fits.sort(key=lambda x: -x[0])
    best = fits[0][1]
    chis = list(dict.fromkeys(round(float(lam[2]), 2) for _, lam in fits))[:MAX_STARTS]
    solutions = profile_cell(model, data, tuple(chis), "aligned", MAP_Q) + profile_cell(model, data, (), "lensed", MAP_Q)
    sol = pd.DataFrame(solutions).assign(**cell)
    misfit = 0.5 * REFERENCE_SNR**2 - sol[sol["kind"] == "lensed"]["loglike"].max()
    best_row = {f"best_{k}": v for k, v in summary_of(best, model.truth).items()}
    best_row["best_ff"] = model.fitting_factor(data, np.exp(best[0]), best[1], best[2], best[2])
    rows = [{**cell, "snr": snr, "mchirp_true": float(np.exp(model.truth[0])), **post, "lens_misfit": misfit, **best_row}
            for snr, post in posteriors(sol, misfit)]  # fmt: skip
    return rows, sol.to_dict("records")


def posteriors(sol: pd.DataFrame, misfit: float) -> list[tuple[float, dict]]:
    """Aligned posterior and ln B(lensed vs aligned) at every SNR, from the fixed-q solutions of one cell."""
    aligned_all, lensed_all = sol[sol["kind"] == "aligned"], sol[sol["kind"] == "lensed"]
    out = []
    for snr in SNRS:
        s = snr / REFERENCE_SNR
        aligned = aligned_all.assign(log_z_s=[log_z_from(r, s) for r in aligned_all.to_dict("records")])
        lensed = lensed_all.assign(log_z_s=[log_z_from(r, s) for r in lensed_all.to_dict("records")])
        post = laplace_posterior(aligned, "log_z_s", var_scale=1.0 / s**2)
        lnb = log_evidence(lensed, "log_z_s", var_scale=1.0 / s**2) - post.pop("log_z")
        out.append((snr, {**post, "lnB_lensed_vs_aligned": lnb, "lnB_perfect_lens": lnb + s**2 * misfit}))
    return out


def rederive_cell(task: tuple) -> list[dict]:
    old, sol = (pd.DataFrame(x) for x in task)
    old = old.set_index("snr")
    return [{**old.loc[snr].to_dict(), "snr": snr, **post} for snr, post in posteriors(sol, old["lens_misfit"].iloc[0])]


def rederive(outdir: Path, workers: int) -> None:
    """Rebuild cells.csv from the stored solutions.csv, without refitting."""
    summary_path = outdir / "cells.csv"
    old, sol = pd.read_csv(summary_path), pd.read_csv(outdir / "solutions.csv")
    keys = [k for k in (*KEYS, "chi_eff_true") if k in old.columns]
    solutions = {k: g.to_dict("list") for k, g in sol.groupby(keys)}
    tasks = [(g.to_dict("list"), solutions[k]) for k, g in old.groupby(keys)]
    with ProcessPoolExecutor(workers) as pool:
        rows = [r for cell_rows in pool.map(rederive_cell, tasks, chunksize=8) for r in cell_rows]
    summary_path.rename(outdir / "cells_before_rederive.csv")
    pd.DataFrame(rows)[list(old.columns)].to_csv(summary_path, index=False)
    print(f"{len(tasks)} cells rederived", flush=True)


def append(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("results/maps"))
    parser.add_argument("--workers", type=int, default=150)
    parser.add_argument("--limit", type=int, help="run only the first N cells (timing)")
    parser.add_argument("--masses", type=float, nargs="*", default=list(TOTAL_MASSES))
    parser.add_argument("--dt-step", type=float, default=1.0, help="keep every n-th delay of DT_MS")
    parser.add_argument("--dense", action="store_true", help="use DENSE_FLUXES instead of FLUXES")
    parser.add_argument("--optics", nargs="*", choices=OPTICS, default=list(OPTICS))
    parser.add_argument("--rederive", action="store_true", help="rebuild cells.csv from solutions.csv")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.rederive:
        rederive(args.outdir, args.workers)
        return
    summary_path, solutions_path = args.outdir / "cells.csv", args.outdir / "solutions.csv"
    done = set()
    if summary_path.exists():
        done = {tuple(r) for r in pd.read_csv(summary_path)[list(KEYS)].drop_duplicates().itertuples(index=False)}
    grid = cells(tuple(args.masses), args.dt_step, DENSE_FLUXES if args.dense else FLUXES, tuple(args.optics))
    todo = [c for c in grid if tuple(c[k] for k in KEYS) not in done][: args.limit]
    print(f"{len(todo)} cells to run ({len(done)} already done)", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(run_cell, c): c for c in todo}
        for n, future in enumerate(as_completed(futures), 1):
            cell = futures[future]
            try:
                rows, solutions = future.result()
            except Exception as error:
                print(f"FAILED {cell}: {error!r}", flush=True)
                continue
            append(summary_path, rows)
            append(solutions_path, solutions)
            if n % 50 == 0 or n == len(todo):
                print(f"{n}/{len(todo)} cells done", flush=True)


if __name__ == "__main__":
    main()
