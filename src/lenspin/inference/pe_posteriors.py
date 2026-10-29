import argparse
import json
from pathlib import Path

import bilby
import numpy as np
import pandas as pd

PERCENTILES = (5, 50, 95)
LEGACY = {"m1": 100.0, "m2": 80.0, "dt": 0.02, "flux": 0.3}


def spins(post: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    q = post["mass_ratio"].to_numpy()
    if "a_1" in post:
        a1, a2, t1, t2 = (post[k].to_numpy() for k in ("a_1", "a_2", "tilt_1", "tilt_2"))
        chi_eff = (a1 * np.cos(t1) + q * a2 * np.cos(t2)) / (1 + q)
        chi_p = np.maximum(a1 * np.sin(t1), (4 * q + 3) / (4 + 3 * q) * q * a2 * np.sin(t2))
        return chi_eff, chi_p
    chi_eff = (post["chi_1"].to_numpy() + q * post["chi_2"].to_numpy()) / (1 + q)
    return chi_eff, np.zeros_like(q)


def summarise(outdir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(outdir.glob("inj*_*.json")):
        meta = {**LEGACY, **json.loads(path.read_text())}
        result = bilby.result.read_in_result(str(outdir / path.stem / f"{path.stem}_result.hdf5"))
        post = result.posterior
        chi_eff, chi_p = spins(post)
        row = {k: meta[k] for k in ("index", "m1", "m2", "dt", "flux", "cosi", "network", "model")}
        row.update(network_snr=meta["network_snr"], log_evidence=meta["log_evidence"])
        row["mchirp_true"] = (meta["m1"] * meta["m2"]) ** 0.6 / (meta["m1"] + meta["m2"]) ** 0.2
        row["q_true"] = meta["m2"] / meta["m1"]
        for name, values in (
            ("mchirp", post["chirp_mass"].to_numpy()),
            ("q", post["mass_ratio"].to_numpy()),
            ("chi_eff", chi_eff),
            ("chi_p", chi_p),
        ):
            lo, mid, hi = np.percentile(values, PERCENTILES)
            row.update({f"{name}_lo": lo, f"{name}_med": mid, f"{name}_hi": hi})
        row["p_chi_eff_negative"] = float(np.mean(chi_eff < 0))
        row["mchirp_at_prior_edge"] = float(
            np.mean(post["chirp_mass"] < result.priors["chirp_mass"].minimum * 1.02)
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    df = summarise(args.outdir)
    if df.empty:
        print("no finished runs")
        return
    lnz = df.pivot_table(index="index", columns="model", values="log_evidence")
    if {"precessing", "aligned"} <= set(lnz.columns):
        df = df.join((lnz["precessing"] - lnz["aligned"]).rename("lnB_prec_vs_aligned"), on="index")
    df.to_csv(args.outdir / "posteriors.csv", index=False)
    cols = ["index", "m1", "m2", "dt", "flux", "model", "network_snr", "mchirp_med", "q_lo", "q_med", "q_hi"]
    cols += [
        "chi_eff_lo",
        "chi_eff_med",
        "chi_eff_hi",
        "chi_p_med",
        "p_chi_eff_negative",
        "mchirp_at_prior_edge",
    ]
    cols += [c for c in ("lnB_prec_vs_aligned",) if c in df]
    print(df.sort_values(["index", "model"])[cols].to_string(index=False, float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    main()
