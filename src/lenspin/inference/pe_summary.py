import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PAIRS = {
    "lnB_precessing_vs_aligned": ("precessing", "aligned"),
    "lnB_lensed_vs_precessing": ("lensed", "precessing"),
    "lnB_lensed_vs_aligned": ("lensed", "aligned"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("results/pe"))
    args = parser.parse_args()
    runs = pd.DataFrame([json.loads(p.read_text()) for p in sorted(args.outdir.glob("inj*_*.json"))])
    if runs.empty:
        print("no finished runs")
        return
    keys = ["index", "cosi", "isky", "network"]
    lnz = runs.pivot_table(index=keys, columns="model", values="log_evidence")
    err = runs.pivot_table(index=keys, columns="model", values="log_evidence_err")
    snr = runs.groupby(keys)["network_snr"].first()
    table = pd.DataFrame({"network_snr": snr})
    for name, (a, b) in PAIRS.items():
        if a in lnz and b in lnz:
            table[name] = lnz[a] - lnz[b]
            table[f"{name}_err"] = np.hypot(err[a], err[b])
    for model in lnz.columns:
        table[f"lnZ_{model}"] = lnz[model]
    table = table.reset_index().sort_values(["cosi", "isky", "network"])
    table.to_csv(args.outdir / "summary.csv", index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\n{len(runs)} finished runs")


if __name__ == "__main__":
    main()
