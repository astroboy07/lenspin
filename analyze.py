import argparse
from pathlib import Path

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
FIG_WIDTH = 7.1
FIG_HEIGHT = 2.6


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(absorbed=df["ff_precessing"] - df["ff_aligned"])
    ref = df[df["network"] == "HL"].set_index([*CONFIG, "isky"])["rho_crit_precessing"]
    df = df.join(ref.rename("rho_crit_hl"), on=[*CONFIG, "isky"])
    df["ratio_to_hl"] = df["rho_crit_precessing"] / df["rho_crit_hl"]
    grouped = df.groupby([*CONFIG, "network"])
    return grouped.agg(
        rho_crit_median=("rho_crit_precessing", "median"),
        rho_crit_p10=("rho_crit_precessing", lambda s: s.quantile(0.1)),
        rho_crit_p90=("rho_crit_precessing", lambda s: s.quantile(0.9)),
        ratio_to_hl_median=("ratio_to_hl", "median"),
        ff_aligned_median=("ff_aligned", "median"),
        ff_precessing_median=("ff_precessing", "median"),
        absorbed_median=("absorbed", "median"),
        snr_dref_median=("snr_dref", "median"),
        d_crit_mpc_median=("d_crit_mpc", "median"),
        n_fringes=("n_fringes", "median"),
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
    panels = table[["dt_ms", "flux_ratio"]].drop_duplicates().sort_values(["dt_ms", "flux_ratio"])
    fig, axes = plt.subplots(1, len(panels), figsize=(FIG_WIDTH, FIG_HEIGHT), sharey=True, squeeze=False)
    for ax, (_, panel) in zip(axes[0], panels.iterrows()):
        sub = table[(table["dt_ms"] == panel["dt_ms"]) & (table["flux_ratio"] == panel["flux_ratio"])]
        for net in NETWORK_ORDER:
            rows = sub[sub["network"] == net].sort_values("cosi")
            if rows.empty:
                continue
            color, marker, ls = STYLE[net]
            ax.plot(rows["cosi"], rows["rho_crit_median"], ls, marker=marker, color=color, label=net)
        ax.set_yscale("log")
        ax.set_title(f"$\\Delta t$ = {panel['dt_ms']:g} ms, $I$ = {panel['flux_ratio']:g}", fontsize=10)
        ax.set_xlabel(r"$\cos\iota$")
    axes[0][0].set_ylabel(r"$\rho_{\rm crit}$ (median over sky)")
    axes[0][-1].legend(frameon=False, loc="best")
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.18, top=0.88, wspace=0.08)
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
        plot(sub, outdir / f"{args.csv.stem}_rho_crit_{sky_mode}")
    cols = [
        "dt_ms",
        "flux_ratio",
        "cosi",
        "network",
        "n_fringes",
        "ff_aligned_median",
        "ff_precessing_median",
    ]
    cols += ["absorbed_median", "rho_crit_median", "ratio_to_hl_median", "d_crit_mpc_median", "n_sky"]
    print(table[cols].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(
        f"\n{len(df)} rows; slowest task {df['seconds'].max():.0f} s, median {df['seconds'].median():.0f} s"
    )


if __name__ == "__main__":
    main()
