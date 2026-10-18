import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from analyze import FIG_WIDTH, NETWORK_ORDER, PANEL_HEIGHT, RHO_CAP, STYLE

CONFIG = ["m1", "m2", "chi1", "cos_tilt1", "chi2z", "cos_theta_jn", "phi_jl", "sky_mode"]


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(gain=df["ff_lensed"] - df["ff_aligned"])
    ref = df[df["network"] == "HL"].set_index([*CONFIG, "isky"])["rho_false_lensing"]
    df = df.join(ref.rename("rho_false_hl"), on=[*CONFIG, "isky"])
    df["false_ratio_to_hl"] = df["rho_false_lensing"] / df["rho_false_hl"]
    grouped = df.groupby([*CONFIG, "network"])
    return grouped.agg(
        chi_perp=("chi_perp", "median"),
        ff_aligned_median=("ff_aligned", "median"),
        ff_lensed_median=("ff_lensed", "median"),
        gain_median=("gain", "median"),
        rho_false_lensing_median=("rho_false_lensing", "median"),
        false_ratio_to_hl_median=("false_ratio_to_hl", "median"),
        rho_crit_lensing_median=("rho_crit_lensing", "median"),
        best_dt_ms_median=("best_dt_ms", "median"),
        best_flux_ratio_median=("best_flux_ratio", "median"),
        snr_dref_median=("snr_dref", "median"),
        n_sky=("isky", "count"),
    ).reset_index()


def plot(table: pd.DataFrame, outbase: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.5,
            "lines.markersize": 4,
            "pdf.fonttype": 42,
        },
    )
    sources = table[["m1", "m2"]].drop_duplicates().sort_values(["m1", "m2"]).to_numpy()
    chis = sorted(table["chi1"].unique())
    fig, axes = plt.subplots(
        len(sources),
        len(chis),
        figsize=(FIG_WIDTH, PANEL_HEIGHT * len(sources) + 0.6),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for i, (m1, m2) in enumerate(sources):
        for j, chi in enumerate(chis):
            ax = axes[i][j]
            sub = table[(table["m1"] == m1) & (table["m2"] == m2) & (table["chi1"] == chi)]
            for net in NETWORK_ORDER:
                rows = sub[sub["network"] == net].sort_values("cos_theta_jn")
                if rows.empty:
                    continue
                color, marker, ls = STYLE[net]
                values = np.minimum(rows["rho_false_lensing_median"], RHO_CAP)
                ax.plot(rows["cos_theta_jn"], values, ls, marker=marker, color=color, label=net)
            ax.set_yscale("log")
            if i == 0:
                ax.set_title(f"$\\chi_{{1\\perp}}$ = {chi:g}", fontsize=10)
            if j == len(chis) - 1:
                ax.text(1.02, 0.5, f"{m1:g}+{m2:g}", transform=ax.transAxes, rotation=90, va="center")
            if i == len(sources) - 1:
                ax.set_xlabel(r"$\cos\theta_{JN}$")
            if j == 0:
                ax.set_ylabel(r"$\rho_{\rm false}^{\rm lens}$")
    axes[0][-1].legend(frameon=False, loc="best")
    fig.subplots_adjust(left=0.1, right=0.94, bottom=0.12, top=0.92, wspace=0.08, hspace=0.12)
    fig.savefig(outbase.with_suffix(".pdf"))
    fig.savefig(outbase.with_suffix(".png"), dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()
    outdir = args.outdir or args.csv.parent
    df = pd.read_csv(args.csv)
    table = summarise(df)
    table.to_csv(outdir / f"{args.csv.stem}_summary.csv", index=False)
    for sky_mode, sub in table.groupby("sky_mode"):
        plot(sub, outdir / f"{args.csv.stem}_rho_false_lensing_{sky_mode}")
    cols = ["m1", "m2", "chi1", "cos_theta_jn", "network", "ff_aligned_median", "ff_lensed_median"]
    cols += ["gain_median", "rho_false_lensing_median", "false_ratio_to_hl_median", "rho_crit_lensing_median"]
    cols += ["best_dt_ms_median", "best_flux_ratio_median", "n_sky"]
    print(table[cols].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    below = (df["ff_lensed"] < df["ff_aligned"]).sum()
    print(f"\n{len(df)} rows; {below} with FF_lensed < FF_aligned; median {df['seconds'].median():.0f} s")


if __name__ == "__main__":
    main()
