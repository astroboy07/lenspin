import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

NETWORK_ORDER = ["H", "HL", "HLV", "HLVI", "ET"]
STYLE = {
    "H": ("#2a78d6", "o", "--"),
    "HL": ("#2a78d6", "s", "-"),
    "HLV": ("#eb6834", "^", "-"),
    "HLVI": ("#1baf7a", "D", "-"),
    "ET": ("#e87ba4", "v", "-"),
}
CONFIG = ["m1", "m2", "dt_ms", "flux_ratio", "cosi", "sky_mode"]
LN_B = 4.6
RHO_REF = 20.0
RHO_CAP = 1e3
FIG_WIDTH = 7.1
PANEL_HEIGHT = 1.9
METRICS = {
    "rho_false": r"$\rho_{\rm false}$",
    "rho_crit": r"$\rho_{\rm crit}$",
}


def add_metrics(df: pd.DataFrame) -> pd.DataFrame:
    gain = df["ff_precessing"] ** 2 - df["ff_aligned"] ** 2
    loss = 1.0 - df["ff_precessing"] ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        rho_false = np.where(gain > 0, np.sqrt(2 * LN_B / gain), np.inf)
        rho_crit = np.where(loss > 0, np.sqrt(2 * LN_B / loss), np.inf)
    df = df.assign(
        absorbed=df["ff_precessing"] - df["ff_aligned"],
        rho_false=rho_false,
        rho_crit=rho_crit,
        false_lnl_ref=0.5 * RHO_REF**2 * gain,
    )
    ref = df[df["network"] == "HL"].set_index([*CONFIG, "isky"])[["rho_false", "rho_crit"]]
    df = df.join(ref.add_suffix("_hl"), on=[*CONFIG, "isky"])
    return df.assign(
        false_ratio_to_hl=df["rho_false"] / df["rho_false_hl"],
        crit_ratio_to_hl=df["rho_crit"] / df["rho_crit_hl"],
    )


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby([*CONFIG, "network"])
    return grouped.agg(
        n_fringes=("n_fringes", "median"),
        ff_aligned_median=("ff_aligned", "median"),
        ff_precessing_median=("ff_precessing", "median"),
        absorbed_median=("absorbed", "median"),
        false_lnl_ref_median=("false_lnl_ref", "median"),
        rho_false_median=("rho_false", "median"),
        false_ratio_to_hl_median=("false_ratio_to_hl", "median"),
        rho_crit_median=("rho_crit", "median"),
        rho_crit_p10=("rho_crit", lambda s: s.quantile(0.1)),
        rho_crit_p90=("rho_crit", lambda s: s.quantile(0.9)),
        crit_ratio_to_hl_median=("crit_ratio_to_hl", "median"),
        snr_dref_median=("snr_dref", "median"),
        n_sky=("isky", "count"),
    ).reset_index()


def plot(table: pd.DataFrame, metric: str, outbase: Path) -> None:
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
    dts = sorted(table["dt_ms"].unique())
    fluxes = sorted(table["flux_ratio"].unique())
    fig, axes = plt.subplots(
        len(fluxes),
        len(dts),
        figsize=(FIG_WIDTH, PANEL_HEIGHT * len(fluxes) + 0.6),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    column = f"{metric}_median"
    for i, flux in enumerate(fluxes):
        for j, dt in enumerate(dts):
            ax = axes[i][j]
            sub = table[(table["dt_ms"] == dt) & (table["flux_ratio"] == flux)]
            for net in NETWORK_ORDER:
                rows = sub[sub["network"] == net].sort_values("cosi")
                if rows.empty:
                    continue
                color, marker, ls = STYLE[net]
                values = np.minimum(rows[column], RHO_CAP)
                ax.plot(rows["cosi"], values, ls, marker=marker, color=color, label=net)
            ax.set_yscale("log")
            if i == 0:
                fringes = sub["n_fringes"].median()
                ax.set_title(f"$\\Delta t$ = {dt:g} ms ({fringes:.1f} fringes)", fontsize=10)
            if j == len(dts) - 1:
                ax.text(1.02, 0.5, f"$I$ = {flux:g}", transform=ax.transAxes, rotation=90, va="center")
            if i == len(fluxes) - 1:
                ax.set_xlabel(r"$\cos\iota$")
            if j == 0:
                ax.set_ylabel(METRICS[metric])
    axes[0][-1].legend(frameon=False, loc="best")
    fig.subplots_adjust(left=0.08, right=0.95, bottom=0.1, top=0.93, wspace=0.08, hspace=0.12)
    fig.savefig(outbase.with_suffix(".pdf"))
    fig.savefig(outbase.with_suffix(".png"), dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()
    outdir = args.outdir or args.csv.parent
    df = add_metrics(pd.read_csv(args.csv))
    table = summarise(df)
    table.to_csv(outdir / f"{args.csv.stem}_summary.csv", index=False)
    for (m1, m2, sky_mode), sub in table.groupby(["m1", "m2", "sky_mode"]):
        for metric in METRICS:
            plot(sub, metric, outdir / f"{args.csv.stem}_{metric}_{m1:g}_{m2:g}_{sky_mode}")
    cols = ["m1", "m2", "dt_ms", "flux_ratio", "cosi", "network", "n_fringes", "absorbed_median"]
    cols += ["false_lnl_ref_median", "rho_false_median", "false_ratio_to_hl_median"]
    cols += ["rho_crit_median", "crit_ratio_to_hl_median", "n_sky"]
    print(table[cols].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(
        f"\n{len(df)} rows; slowest task {df['seconds'].max():.0f} s, median {df['seconds'].median():.0f} s"
    )
    print(
        f"false_lnl_ref = 0.5 * {RHO_REF:g}^2 * (FF_prec^2 - FF_aligned^2); rho_* capped at {RHO_CAP:g} on plots"
    )


if __name__ == "__main__":
    main()
