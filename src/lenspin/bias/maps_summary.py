import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HIDDEN = (3.0, 8.0)
MTSUN_MS = 4.925491e-3
CELL = ("total_mass", "q_true", "dt_ms", "flux", "optics")
Q_ONE_EDGE = 0.9
DIVERGING = LinearSegmentedColormap.from_list(
    "chi_eff", ["#104281", "#2a78d6", "#9ec5f4", "#f0efec", "#f2b3b2", "#e34948", "#8f1d1d"]
)
SEQUENTIAL = LinearSegmentedColormap.from_list("count", ["#f0efec", "#9ec5f4", "#2a78d6", "#104281"])
INK, MUTED = "#0b0b0b", "#898781"


def flags(cells: pd.DataFrame) -> pd.DataFrame:
    """A cell is biased when the 90% interval excludes the truth in chi_eff or q (q = 1 sits on the prior
    edge, so there the upper bound must fall below Q_ONE_EDGE). Hidden means ln B(lensed vs aligned),
    for a lensed template that fits perfectly, is below the threshold."""
    q_one = cells["q_true"] >= 1.0
    q_biased = np.where(q_one, cells["q_hi"] < Q_ONE_EDGE,
                        (cells["q_lo"] > cells["q_true"]) | (cells["q_hi"] < cells["q_true"]))  # fmt: skip
    out = cells.assign(
        chi_biased=(cells["chi_eff_lo"] > 0) | (cells["chi_eff_hi"] < 0),
        q_biased=q_biased,
        mchirp_biased=(cells["mchirp_lo"] > cells["mchirp_true"]) | (cells["mchirp_hi"] < cells["mchirp_true"]),
    )
    out["biased"] = out["chi_biased"] | out["q_biased"]
    out["dt_over_m"] = out["dt_ms"] / (out["total_mass"] * MTSUN_MS)
    for level in HIDDEN:
        out[f"danger_{level:g}"] = out["biased"] & (out["lnB_perfect_lens"] < level)
    return out


def tables(cells: pd.DataFrame) -> None:
    group = cells.groupby(["optics", "total_mass", "q_true", "snr"])
    summary = group.agg(
        cells=("biased", "size"), biased=("biased", "mean"), chi_biased=("chi_biased", "mean"),
        q_biased=("q_biased", "mean"), danger_3=("danger_3", "mean"), danger_8=("danger_8", "mean"),
    )  # fmt: skip
    with pd.option_context("display.width", 200, "display.max_rows", 400):
        print(summary.round(3).to_string())
    biased = cells[cells["biased"]].groupby(["total_mass", "snr"])
    hidden = pd.DataFrame({"n_biased": biased.size(), "hidden_3": biased["danger_3"].mean(),
                           "hidden_8": biased["danger_8"].mean()})  # fmt: skip
    print("\nFraction of biased cells that are hidden (both optics and true q):")
    print(hidden.round(2).to_string())
    danger = cells[cells["danger_3"]].drop_duplicates(list(CELL))
    print("\nDelay of danger cells in units of G M / c^3, by total mass:")
    print(danger.groupby("total_mass")["dt_over_m"].describe().round(1).to_string())
    danger = cells[cells["danger_3"]].copy()
    danger["shift"] = danger["chi_eff_med"].abs()
    cols = ["optics", "total_mass", "q_true", "dt_ms", "flux", "snr", "chi_eff_lo", "chi_eff_med", "chi_eff_hi",
            "q_lo", "q_med", "q_hi", "mchirp_med", "mchirp_true", "lnB_perfect_lens"]  # fmt: skip
    print("\nStrongest chi_eff shifts with ln B < 3:")
    print(danger.sort_values("shift", ascending=False)[cols].head(15).round(3).to_string(index=False))


def heatmaps(cells: pd.DataFrame, column: str, snr: float, path: Path, title: str, cmap, norm=None) -> None:
    sub = cells[(cells["snr"] == snr) & (cells["optics"] == "geometric")]
    masses, qs = sorted(sub["total_mass"].unique()), sorted(sub["q_true"].unique(), reverse=True)
    fig, axes = plt.subplots(len(qs), len(masses), figsize=(3.0 * len(masses), 2.6 * len(qs)), sharex=True, sharey=True,
                             squeeze=False, constrained_layout=True)  # fmt: skip
    image = None
    for i, q in enumerate(qs):
        for j, m in enumerate(masses):
            ax = axes[i, j]
            grid = sub[(sub["total_mass"] == m) & (sub["q_true"] == q)].pivot(index="flux", columns="dt_ms", values=column)
            hidden = sub[(sub["total_mass"] == m) & (sub["q_true"] == q)].pivot(
                index="flux", columns="dt_ms", values="lnB_perfect_lens"
            )
            if grid.empty:
                ax.set_axis_off()
                continue
            x, y = grid.columns.to_numpy(), np.arange(len(grid.index))
            image = ax.pcolormesh(x, y, grid.to_numpy(float), cmap=cmap, norm=norm, shading="nearest")
            ax.contour(x, y, hidden.to_numpy(float), levels=list(HIDDEN), colors=INK, linewidths=1.0,
                       linestyles=["-", "--"])  # fmt: skip
            ax.set_yticks(y, [f"{v:g}" for v in grid.index])
            ax.set_title(f"M = {m:g}, q = {q:g}", fontsize=9, color=INK)
            for spine in ax.spines.values():
                spine.set_color(MUTED)
            ax.tick_params(colors=MUTED, labelsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("lens delay [ms]", fontsize=9, color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel("flux ratio", fontsize=9, color=INK)
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.8, label=title)
    fig.suptitle(f"{title} at SNR {snr:g} (geometric optics); contours: ln B = 3 (solid), 8 (dashed)",
                 fontsize=10, color=INK)  # fmt: skip
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("results/maps"))
    parser.add_argument("--snr", type=float, default=12.0)
    args = parser.parse_args()
    cells = flags(pd.read_csv(args.outdir / "cells.csv"))
    print(f"{cells[list(CELL)].drop_duplicates().shape[0]} cells")
    tables(cells)
    heatmaps(cells, "chi_eff_med", 25.0, args.outdir / "chi_eff_snr25.png", "chi_eff median", DIVERGING,
             TwoSlopeNorm(0.0, -0.8, 0.8))  # fmt: skip
    counts = cells.groupby(list(CELL))["danger_3"].sum().rename("danger_snrs")
    merged = cells.merge(counts.reset_index(), on=list(CELL))
    heatmaps(merged, "danger_snrs", args.snr, args.outdir / "danger_count.png",
             "SNRs (of 8) biased with ln B < 3", SEQUENTIAL)  # fmt: skip


if __name__ == "__main__":
    main()
