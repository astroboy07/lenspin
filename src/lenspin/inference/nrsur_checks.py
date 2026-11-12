import warnings
from multiprocessing import Pool
from pathlib import Path

import bilby
import numpy as np
import pandas as pd

from lenspin.inference.pe import STUDIES, injection_distance, make_ifos, make_likelihood, make_priors

warnings.filterwarnings("ignore")
bilby.core.utils.logger.setLevel("ERROR")

NRSUR = Path("results/pe_nrsur")
RUNS = {
    10.5: {"NRSur7dq4": (NRSUR, 0), "XP": (Path("results/pe_scan"), 0), "XO4a": (Path("results/pe_waveform"), 1)},
    20.5: {"NRSur7dq4": (NRSUR, 1), "XP": (Path("results/pe_scan"), 4), "XO4a": (Path("results/pe_waveform"), 2)},
}
TILTED = {"chi_eff_max": -0.4, "q_max": 0.35}
EDGE_ON_COS = 0.3
NSAMPLES = 150
SEED = 5
WORKERS = 16
STATE = {}


def posterior(outdir: Path, index: int, model: str) -> pd.DataFrame:
    label = f"inj{index}_{model}"
    return bilby.result.read_in_result(str(outdir / label / f"{label}_result.hdf5")).posterior


def spins(post: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    q = post["mass_ratio"].to_numpy()
    if "a_1" not in post:
        return (post["chi_1"].to_numpy() + q * post["chi_2"].to_numpy()) / (1 + q), np.zeros_like(q)
    a1, a2, t1, t2 = (post[k].to_numpy() for k in ("a_1", "a_2", "tilt_1", "tilt_2"))
    chi_eff = (a1 * np.cos(t1) + q * a2 * np.cos(t2)) / (1 + q)
    chi_p = np.maximum(a1 * np.sin(t1), (4 * q + 3) / (4 + 3 * q) * q * a2 * np.sin(t2))
    return chi_eff, chi_p


def interval(x: np.ndarray) -> str:
    lo, med, hi = np.percentile(x, [5, 50, 95])
    return f"{med:+.2f} [{lo:+.2f},{hi:+.2f}]"


def medians() -> pd.DataFrame:
    rows = []
    for dt, runs in RUNS.items():
        for approx, (outdir, index) in runs.items():
            for model in ("precessing", "aligned"):
                post = posterior(outdir, index, model)
                q = post["mass_ratio"].to_numpy()
                chi_eff, chi_p = spins(post)
                cos_jn = np.abs(np.cos(post["theta_jn"].to_numpy()))
                rows.append({
                    "dt_ms": dt, "approximant": approx, "model": model,
                    "chi_eff": interval(chi_eff), "q": interval(q), "mchirp": interval(post["chirp_mass"].to_numpy()),
                    "chi_p_med": np.median(chi_p), "abs_cos_jn_med": np.median(cos_jn),
                    "frac_tilted": np.mean((chi_eff < TILTED["chi_eff_max"]) & (q < TILTED["q_max"])),
                    "frac_edge_on": np.mean(cos_jn < EDGE_ON_COS), "maxlnl": post["log_likelihood"].max(),
                })  # fmt: skip
    return pd.DataFrame(rows)


KEYS = {
    "precessing": ["chirp_mass", "mass_ratio", "a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl"],
    "aligned": ["chirp_mass", "mass_ratio", "chi_1", "chi_2"],
}
COMMON = ["theta_jn", "psi", "phase", "ra", "dec", "geocent_time", "luminosity_distance"]
NPRIOR = 3000
QUADRUPOLE = [[2, 2], [2, -2]]
NBEST = 50


def setup() -> None:
    for index, inj in enumerate(STUDIES["nrsur"]):
        ifos = make_ifos(inj, injection_distance(inj))
        for model in KEYS:
            priors = make_priors(model, inj)
            lookup = NRSUR / f"inj{index}_{model}" / "distance_lookup.npz"
            STATE[model, index] = make_likelihood(model, inj, ifos, priors, lookup=lookup)
            STATE[model, index, "22"] = make_likelihood(model, {**inj, "mode_array": QUADRUPOLE}, ifos,
                                                        make_priors(model, inj), lookup=lookup)  # fmt: skip


def evaluate(job: tuple) -> float:
    target, params = job
    like = STATE[target]
    like.parameters.update(params)
    return float(like.log_likelihood_ratio())


def pick(post: pd.DataFrame, rng) -> pd.DataFrame:
    draw = post.iloc[rng.choice(len(post), min(NSAMPLES, len(post)), replace=False)]
    return pd.concat([post.loc[[post["log_likelihood"].idxmax()]], draw])


def cross(pool) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    sets = []
    for model in KEYS:
        for dt, runs in RUNS.items():
            target = runs["NRSur7dq4"][1]
            for approx, (outdir, index) in runs.items():
                sets.append((model, dt, approx, target, pick(posterior(outdir, index, model), rng)))
    jobs = [((m, t), {**{k: float(r[k]) for k in KEYS[m] + COMMON}, "time_jitter": float(r.get("time_jitter", 0.0))})
            for m, _, _, t, df in sets for _, r in df.iterrows()]  # fmt: skip
    values = np.array(pool.map(evaluate, jobs, chunksize=2))
    values[values < -1e300] = np.nan
    rows, k = [], 0
    for model, dt, approx, target, df in sets:
        lnl = values[k : k + len(df)]
        k += len(df)
        own = df["log_likelihood"].to_numpy()
        chi_eff, _ = spins(df)
        is_tilted = (chi_eff < TILTED["chi_eff_max"]) & (df["mass_ratio"].to_numpy() < TILTED["q_max"])
        q = df["mass_ratio"].to_numpy()
        rows.append({
            "model": model, "dt_ms": dt, "samples": approx, "n": len(df), "n_failed": int(np.isnan(lnl).sum()),
            "n_q_below_sixth": int((q < 1 / 6).sum()), "lnl_max_q_above_sixth": np.nanmax(np.where(q >= 1 / 6, lnl, np.nan)),
            "self_check_max_dev": float(np.nanmax(np.abs(lnl - own))) if approx == "NRSur7dq4" else np.nan,
            "lnl_at_source_best": lnl[0], "lnl_max": np.nanmax(lnl), "lnl_median": np.nanmedian(lnl),
            "lnl_max_tilted": np.nanmax(lnl[is_tilted]) if is_tilted.any() else np.nan, "n_tilted": int(is_tilted.sum()),
            "nrsur_own_max": posterior(NRSUR, target, model)["log_likelihood"].max(),
        })  # fmt: skip
    df = pd.DataFrame(rows)
    df["deficit_vs_nrsur_max"] = df["nrsur_own_max"] - df["lnl_max"]
    return df


def coverage(pool) -> pd.DataFrame:
    rows = []
    bilby.core.utils.random.seed(SEED)
    for model in KEYS:
        priors = make_priors(model, STUDIES["nrsur"][0])
        draws = pd.DataFrame(priors.sample(NPRIOR))
        jobs = [((model, 0), {**{k: float(r[k]) for k in KEYS[model] + COMMON}, "time_jitter": 0.0})
                for _, r in draws.iterrows()]  # fmt: skip
        failed = np.array(pool.map(evaluate, jobs, chunksize=8)) < -1e300
        q = draws["mass_ratio"].to_numpy()
        rows.append({
            "model": model, "prior_mass_below_sixth": priors["mass_ratio"].cdf(1 / 6), "n": NPRIOR,
            "failed": int(failed.sum()), "q_below_sixth": int((q < 1 / 6).sum()),
            "failed_q_above_sixth": int((failed & (q >= 1 / 6)).sum()), "max_q_failed": q[failed].max() if failed.any() else np.nan,
            "min_q_ok": q[~failed].min(),
        })  # fmt: skip
    return pd.DataFrame(rows)


def modes(pool) -> pd.DataFrame:
    rows = []
    for model in KEYS:
        for index, inj in enumerate(STUDIES["nrsur"]):
            post = posterior(NRSUR, index, model).nlargest(NBEST, "log_likelihood")
            jobs = [((model, index, "22"), {**{k: float(r[k]) for k in KEYS[model] + COMMON},
                     "time_jitter": float(r.get("time_jitter", 0.0))}) for _, r in post.iterrows()]  # fmt: skip
            drop = post["log_likelihood"].to_numpy() - np.array(pool.map(evaluate, jobs, chunksize=2))
            rows.append({
                "model": model, "dt_ms": round(inj["dt"] * 1e3, 1), "n": len(post), "drop_at_best": drop[0],
                "drop_median": np.median(drop), "drop_min": drop.min(), "drop_max": drop.max(),
                "abs_cos_jn_best": abs(np.cos(post["theta_jn"].iloc[0])),
            })  # fmt: skip
    return pd.DataFrame(rows)


def main() -> None:
    med = medians()
    med.to_csv(NRSUR / "routes.csv", index=False)
    print(med.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    with Pool(WORKERS, initializer=setup) as pool:
        df = cross(pool)
        cov = coverage(pool)
        mod = modes(pool)
    df.to_csv(NRSUR / "cross_likelihood.csv", index=False)
    cov.to_csv(NRSUR / "coverage.csv", index=False)
    mod.to_csv(NRSUR / "quadrupole_only.csv", index=False)
    print()
    print(df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print()
    print(cov.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print(mod.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
