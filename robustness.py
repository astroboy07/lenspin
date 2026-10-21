import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from maps import REFERENCE_SNR, injection
from pe import APPROXIMANT, STUDIES
from pe_posteriors import summarise
from predictor import Model, log_z_from

BASELINES = ("danger", "scan")
ROBUSTNESS = ("noise_danger", "noise_scan", "inclination", "waveform")
VARIED = ("cosi", "approximant", "noise_seed")
SHOW = ("chi_eff", "q", "mchirp")
SCAN_CELL = {"total_mass": 180.0, "q_true": 1.0, "flux": 0.35, "optics": "geometric"}
SCAN_DT_MS = (10.5, 12.5, 14.5, 16.5, 20.5)
SCAN_SNR = 25.0


def solution_lam(r: dict) -> np.ndarray:
    return np.array([r["log_mchirp"], r["q"] / (1 + r["q"]) ** 2, r["chi_eff"], r["tc"], r["phic"], r["log_amp"]])


def interval(r: dict, model: str, key: str) -> str:
    return f"{r[f'{model}_{key}_med']:+.2f} [{r[f'{model}_{key}_lo']:+.2f},{r[f'{model}_{key}_hi']:+.2f}]"


def baseline_of(inj: dict) -> tuple[str, int]:
    for study in BASELINES:
        for index, base in enumerate(STUDIES[study]):
            if all(inj.get(k) == v for k, v in base.items() if k not in VARIED):
                return study, index
    raise ValueError(f"no C2 baseline for {inj}")


def table(study: str, outdir: Path) -> pd.DataFrame:
    pe = summarise(outdir)
    lnz = pe.pivot_table(index="index", columns="model", values="log_evidence")
    rows = []
    for index, inj in enumerate(STUDIES[study]):
        base = (study, index) if study in BASELINES else baseline_of(inj)
        row = {"study": study, "index": index, "base_study": base[0], "base_index": base[1], "cosi": inj["cosi"],
               "approximant": inj.get("approximant", APPROXIMANT), "noise_seed": inj.get("noise_seed")}  # fmt: skip
        for r in pe[pe["index"] == index].to_dict("records"):
            row.update({f"{r['model']}_{k}_{s}": r[f"{k}_{s}"] for k in SHOW for s in ("lo", "med", "hi")})
        for model in ("precessing", "lensed"):
            if {model, "aligned"} <= set(lnz.columns) and index in lnz.index:
                row[f"lnB_{model}_vs_aligned"] = lnz.loc[index, model] - lnz.loc[index, "aligned"]
        rows.append(row)
    return pd.DataFrame(rows)


def mode_competition(solutions: pd.DataFrame) -> pd.DataFrame:
    """Evidence gap between the best low-q and high-q aligned modes of the predictor, and the Gaussian-noise
    scatter of that gap, sqrt(<h_a - h_b|h_a - h_b>), at each scan delay."""
    s = SCAN_SNR / REFERENCE_SNR
    rows = []
    for dt in SCAN_DT_MS:
        cell = {**SCAN_CELL, "dt_ms": dt}
        g = solutions[(solutions["kind"] == "aligned") & (solutions["optics"] == cell["optics"])]
        for k in ("total_mass", "q_true", "flux", "dt_ms"):
            g = g[np.isclose(g[k], cell[k])]
        g = g.assign(log_z_s=[log_z_from(r, s) for r in g.to_dict("records")])
        best = [b.loc[b["log_z_s"].idxmax()] for b in (g[g["q"] < 0.5], g[g["q"] >= 0.5])]
        a, b = sorted(best, key=lambda r: -r["log_z_s"])
        model = Model(injection(cell), "xp", REFERENCE_SNR, cell["optics"])
        diff = model.template(solution_lam(a)) - model.template(solution_lam(b))
        gap, sigma = a["log_z_s"] - b["log_z_s"], s * float(np.sqrt(model.inner(diff, diff)))
        rows.append({"dt_ms": dt, "snr": SCAN_SNR, "q_a": a["q"], "chi_eff_a": a["chi_eff"], "q_b": b["q"],
                     "chi_eff_b": b["chi_eff"], "gap": gap, "sigma_noise": sigma, "p_flip": norm.sf(gap / sigma)})  # fmt: skip
    return pd.DataFrame(rows)


def show(runs: pd.DataFrame) -> None:
    for (study, index), group in runs.groupby(["base_study", "base_index"], sort=False):
        inj = STUDIES[study][index]
        print(f"M{inj['m1'] + inj['m2']:.0f} q{inj['m2'] / inj['m1']:.2f} dt {inj['dt'] * 1e3:.1f} ms "
              f"I {inj['flux']} SNR {inj['target_snr']:g}")  # fmt: skip
        for r in group.sort_values("study", key=lambda c: c != study).to_dict("records"):
            tag = "C2" if r["study"] in BASELINES else f"{r['study']} cosi {r['cosi']} {r['approximant']} seed {r['noise_seed']}"
            parts = [f"{m[:4]} chi_eff {interval(r, m, 'chi_eff')} q {interval(r, m, 'q')} Mc {r[f'{m}_mchirp_med']:.1f}"
                     for m in ("aligned", "lensed", "precessing") if pd.notna(r.get(f"{m}_chi_eff_med"))]  # fmt: skip
            lnb = [f"lnB_{m[:4]} {r[f'lnB_{m}_vs_aligned']:.1f}" for m in ("lensed", "precessing")
                   if pd.notna(r.get(f"lnB_{m}_vs_aligned"))]  # fmt: skip
            print(f"  {tag:48s} " + " | ".join(parts) + "  " + " ".join(lnb))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solutions", type=Path, default=Path("results/maps/solutions.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/robustness"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runs = pd.concat([table(study, Path(f"results/pe_{study}")) for study in BASELINES + ROBUSTNESS])
    runs = runs[runs["study"].isin(ROBUSTNESS) | runs.set_index(["study", "index"]).index.isin(
        runs[runs["study"].isin(ROBUSTNESS)].set_index(["base_study", "base_index"]).index)]  # fmt: skip
    runs.to_csv(args.out / "runs.csv", index=False)
    show(runs)
    modes = mode_competition(pd.read_csv(args.solutions))
    modes.to_csv(args.out / "mode_competition.csv", index=False)
    print(modes.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
