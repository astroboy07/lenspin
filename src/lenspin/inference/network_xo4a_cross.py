import warnings
from multiprocessing import Pool
from pathlib import Path

import bilby
import numpy as np
import pandas as pd

from lenspin.inference.pe import STUDIES, injection_distance, make_ifos, make_likelihood, make_priors

warnings.filterwarnings("ignore")
bilby.core.utils.logger.setLevel("ERROR")

XO4A = Path("results/pe_network_xo4a")
XP = Path("results/pe_network_wide")
PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]
NSAMPLES = 150
SEED = 3
WORKERS = 8
STATE = {}


def setup() -> None:
    for index, inj in enumerate(STUDIES["network_xo4a"]):
        ifos = make_ifos(inj, injection_distance(inj))
        priors = make_priors("precessing", inj)
        lookup = XO4A / f"inj{index}_precessing" / "distance_lookup.npz"
        STATE[index] = make_likelihood("precessing", inj, ifos, priors, lookup=lookup)


def evaluate(job: tuple) -> float:
    target, params = job
    like = STATE[target]
    like.parameters.update(params)
    return float(like.log_likelihood_ratio())


def samples(outdir: Path, index: int, rng) -> pd.DataFrame:
    label = f"inj{index}_precessing"
    post = bilby.result.read_in_result(str(outdir / label / f"{label}_result.hdf5")).posterior
    pick = post.iloc[rng.choice(len(post), min(NSAMPLES, len(post)), replace=False)]
    return pd.concat([post.loc[[post["log_likelihood"].idxmax()]], pick])


def main() -> None:
    rng = np.random.default_rng(SEED)
    sets = []
    for hl, hlvi in PAIRS:
        for target, source in ((hl, hl), (hlvi, hlvi), (hlvi, hl), (hl, hlvi)):
            sets.append(("XO4a", source, target, samples(XO4A, source, rng)))
        for target in (hl, hlvi):
            sets.append(("XP", target, target, samples(XP, target, rng)))
    keys = ["chirp_mass", "mass_ratio", "a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl", "theta_jn", "psi",
            "phase", "ra", "dec", "geocent_time", "luminosity_distance"]  # fmt: skip
    jobs = [(t, {**{k: float(r[k]) for k in keys}, "time_jitter": float(r.get("time_jitter", 0.0))})
            for _, _, t, df in sets for _, r in df.iterrows()]  # fmt: skip
    with Pool(WORKERS, initializer=setup) as pool:
        values = pool.map(evaluate, jobs, chunksize=4)
    rows, k = [], 0
    for approx, source, target, df in sets:
        n = len(df)
        lnl = np.array(values[k : k + n])
        k += n
        own = df["log_likelihood"].to_numpy()
        rows.append({
            "samples": approx, "source": source, "target": target, "n": n,
            "self_check_max_dev": float(np.abs(lnl - own).max()) if approx == "XO4a" and source == target else np.nan,
            "lnl_at_source_best": lnl[0], "lnl_max": lnl.max(), "lnl_median": np.median(lnl),
            "target_own_max": float(bilby.result.read_in_result(
                str(XO4A / f"inj{target}_precessing" / f"inj{target}_precessing_result.hdf5")
            ).posterior["log_likelihood"].max()),
        })  # fmt: skip
    df = pd.DataFrame(rows)
    df["deficit_vs_target_max"] = df["target_own_max"] - df["lnl_max"]
    df.to_csv(XO4A / "cross_likelihood.csv", index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
