import sys
from pathlib import Path

import bilby
import numpy as np
import pandas as pd

STUDIES = {"XP": Path("results/pe_network_wide"), "XO4a": Path("results/pe_network_xo4a")}
TILTED = {"chi_eff_max": -0.4, "q_max": 0.35}
EDGE_ON_COS = 0.3


def law_fits(tables: dict) -> None:
    print("lnB_p/a = a rho^2 - c, least squares per inclination")
    for name, t in tables.items():
        for cosi, g in t.groupby("cosi"):
            x, y = g["network_snr"].to_numpy() ** 2, g["lnB_precessing_vs_aligned"].to_numpy()
            a, b = np.polyfit(x, y, 1)
            res = y - (a * x + b)
            print(f"  {name:5s} cos {cosi}: a {a:.4f} c {-b:5.1f} max|res| {np.abs(res).max():4.1f}")
            for net, h in g.groupby("network"):
                hx, hy = h["network_snr"].to_numpy() ** 2, h["lnB_precessing_vs_aligned"].to_numpy()
                print(f"      {net:4s} only: lnB/rho^2 = {np.round(hy / hx, 4)}")
    print("\nlnB_p/a / rho^2 at the sky where HL and HLVI reach the same SNR (isky 0)")
    for name, t in tables.items():
        for cosi, g in t[t["isky"] == 0].groupby("cosi"):
            r = g.set_index("network")
            v = r["lnB_precessing_vs_aligned"] / r["network_snr"] ** 2
            print(
                f"  {name:5s} cos {cosi}: HL {v['HL']:.4f} HLVI {v['HLVI']:.4f} ratio {v['HLVI'] / v['HL']:.2f}"
                f"  (rho {r.loc['HL', 'network_snr']:.1f} / {r.loc['HLVI', 'network_snr']:.1f})"
            )


def medians(tables: dict) -> pd.DataFrame:
    rows = []
    for name, outdir in STUDIES.items():
        t = tables[name].set_index("index")
        for model in ("precessing", "aligned"):
            for index in t.index:
                label = f"inj{index}_{model}"
                post = bilby.result.read_in_result(str(outdir / label / f"{label}_result.hdf5")).posterior
                q = post["mass_ratio"].to_numpy()
                if "a_1" in post:
                    a1, a2, t1, t2 = (post[k].to_numpy() for k in ("a_1", "a_2", "tilt_1", "tilt_2"))
                    chi_eff = (a1 * np.cos(t1) + q * a2 * np.cos(t2)) / (1 + q)
                    chi_p = np.maximum(a1 * np.sin(t1), (4 * q + 3) / (4 + 3 * q) * q * a2 * np.sin(t2))
                else:
                    chi_eff = (post["chi_1"].to_numpy() + q * post["chi_2"].to_numpy()) / (1 + q)
                    chi_p = np.zeros_like(q)
                cos_jn = np.cos(post["theta_jn"].to_numpy())
                best = post["log_likelihood"].idxmax()
                rows.append({
                    "approximant": name, "model": model, "index": index, "network": t.loc[index, "network"],
                    "cosi": t.loc[index, "cosi"], "isky": t.loc[index, "isky"], "snr": t.loc[index, "network_snr"],
                    "mchirp_med": np.median(post["chirp_mass"]), "q_med": np.median(q),
                    "chi_eff_med": np.median(chi_eff), "chi_p_med": np.median(chi_p),
                    "abs_cos_jn_med": np.median(np.abs(cos_jn)), "theta_jn_med": np.median(post["theta_jn"]),
                    "frac_tilted": np.mean((chi_eff < TILTED["chi_eff_max"]) & (q < TILTED["q_max"])),
                    "frac_edge_on": np.mean(np.abs(cos_jn) < EDGE_ON_COS),
                    "best_mchirp": post["chirp_mass"][best], "best_q": q[best], "best_chi_eff": chi_eff[best],
                    "best_abs_cos_jn": abs(cos_jn[best]), "maxlnl": post["log_likelihood"].max(),
                })  # fmt: skip
    return pd.DataFrame(rows)


def main() -> None:
    tables = {name: pd.read_csv(d / "summary.csv") for name, d in STUDIES.items()}
    law_fits(tables)
    df = medians(tables)
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else STUDIES["XO4a"] / "routes.csv"
    df.to_csv(out, index=False)
    cols = ["approximant", "index", "network", "cosi", "isky", "snr", "mchirp_med", "q_med", "chi_eff_med",
            "chi_p_med", "abs_cos_jn_med", "frac_tilted", "frac_edge_on", "best_q", "best_chi_eff",
            "best_abs_cos_jn", "maxlnl"]  # fmt: skip
    prec = df[df["model"] == "precessing"].sort_values(["cosi", "isky", "network", "approximant"])
    print("\nprecessing posteriors (true Mc 77.8, q 0.80, chi_eff 0)")
    print(prec[cols].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    al = df[df["model"] == "aligned"].sort_values(["cosi", "isky", "network", "approximant"])
    print("\naligned posteriors")
    print(al[cols[:9] + ["maxlnl"]].to_string(index=False, float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    main()
