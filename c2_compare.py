import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pe import STUDIES
from pe_posteriors import summarise

KEYS = ("total_mass", "q_true", "dt_ms", "flux", "optics")
SHOW = ("chi_eff", "q", "mchirp")


def cell_of(inj: dict) -> dict:
    m1, m2 = inj["m1"], inj["m2"]
    return {"total_mass": round(m1 + m2, 3), "q_true": round(m2 / m1, 3), "dt_ms": round(inj["dt"] * 1e3, 3),
            "flux": inj["flux"], "optics": inj.get("optics", "geometric")}  # fmt: skip


def compare(study: str, outdir: Path, maps: pd.DataFrame) -> pd.DataFrame:
    pe = summarise(outdir)
    lnz = pe.pivot_table(index="index", columns="model", values="log_evidence")
    rows = []
    for index, inj in enumerate(STUDIES[study]):
        cell = cell_of(inj)
        pred = maps
        for k, v in cell.items():
            pred = pred[np.isclose(pred[k], v)] if k != "optics" else pred[pred[k] == v]
        pred = pred[np.isclose(pred["snr"], inj["target_snr"])]
        row = {"index": index, **cell, "snr": inj["target_snr"]}
        if len(pred) == 1:
            p = pred.iloc[0]
            row.update({f"map_{k}_{s}": p[f"{k}_{s}"] for k in SHOW for s in ("lo", "med", "hi")})
            row.update(map_lnB=p["lnB_lensed_vs_aligned"], map_lnB_perfect=p["lnB_perfect_lens"])
        for model in ("aligned", "precessing"):
            r = pe[(pe["index"] == index) & (pe["model"] == model)]
            if len(r):
                r = r.iloc[0]
                row.update({f"{model}_{k}_{s}": r[f"{k}_{s}"] for k in SHOW for s in ("lo", "med", "hi")})
        r = pe[(pe["index"] == index) & (pe["model"] == "precessing")]
        if len(r):
            row["precessing_chi_p_med"] = r.iloc[0]["chi_p_med"]
        if {"precessing", "aligned"} <= set(lnz.columns):
            row["lnB_prec_vs_aligned"] = lnz.loc[index, "precessing"] - lnz.loc[index, "aligned"]
        if "lensed" in lnz.columns:
            row["lnB_lensed_vs_aligned"] = lnz.loc[index, "lensed"] - lnz.loc[index, "aligned"]
        rows.append(row)
    return pd.DataFrame(rows)


def show(table: pd.DataFrame) -> None:
    fmt = lambda v: f"{v:.3g}"
    for _, r in table.iterrows():
        head = f"#{int(r['index'])} M{r['total_mass']:g} q{r['q_true']:g} dt {r['dt_ms']:g} ms I {r['flux']:g} {r['optics']} SNR {r['snr']:g}"
        print(head)
        for k in SHOW:
            parts = []
            for src in ("map", "aligned", "precessing"):
                if f"{src}_{k}_med" in r and pd.notna(r[f"{src}_{k}_med"]):
                    parts.append(f"{src} {fmt(r[f'{src}_{k}_med'])} [{fmt(r[f'{src}_{k}_lo'])}, {fmt(r[f'{src}_{k}_hi'])}]")
            print(f"   {k:8s} " + " | ".join(parts))
        extra = [f"{k} {fmt(r[k])}" for k in ("map_lnB", "map_lnB_perfect", "lnB_lensed_vs_aligned",
                                              "lnB_prec_vs_aligned", "precessing_chi_p_med") if k in r and pd.notna(r[k])]  # fmt: skip
        print("   " + ", ".join(extra))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps", type=Path, default=Path("results/maps/cells.csv"))
    args = parser.parse_args()
    maps = pd.read_csv(args.maps)
    for study, outdir in (("danger", Path("results/pe_danger")), ("scan", Path("results/pe_scan"))):
        table = compare(study, outdir, maps)
        table.to_csv(outdir / "c2_compare.csv", index=False)
        print(f"===== {study}")
        show(table)


if __name__ == "__main__":
    main()
