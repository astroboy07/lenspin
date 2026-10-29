import argparse
from pathlib import Path

import bilby
import lal
import lalsimulation
import numpy as np
import pandas as pd

from lenspin.inference.pe import STUDIES, injection_distance, make_ifos
from lenspin.models.signals import F_REF, component_masses

NSAMPLES = 400
NFREQ = 200
SEED = 1
BAND = (0.05, 0.95)


def snr_band(inj: dict) -> dict:
    """SNR-weighted frequency stats of the unlensed signal, and the lens fringes inside that band."""
    ifos = make_ifos({**inj, "flux": 0.0}, injection_distance(inj))
    mask = ifos[0].frequency_mask
    f = ifos[0].frequency_array[mask]
    w = sum(np.abs(ifo.frequency_domain_strain[mask]) ** 2 / ifo.power_spectral_density_array[mask] for ifo in ifos)
    cdf = np.cumsum(w) / w.sum()
    f_lo, f_mid, f_hi = np.interp([BAND[0], 0.5, BAND[1]], cdf, f)
    f_star = float((f * w).sum() / w.sum())
    return {
        "f_star": f_star,
        "f_lo": f_lo,
        "f_mid": f_mid,
        "f_hi": f_hi,
        "fringes_band": inj["dt"] * (f_hi - f_lo),
        "fringe_phase": (f_star * inj["dt"]) % 1.0,
        "sqrt_flux": np.sqrt(inj["flux"]),
    }


def angles(row: pd.Series, freqs: np.ndarray, method) -> tuple[np.ndarray, np.ndarray]:
    m1, m2 = component_masses(row["chirp_mass"], row["mass_ratio"])
    iota, *spins = bilby.gw.conversion.bilby_to_lalsimulation_spins(
        row["theta_jn"], row["phi_jl"], row["tilt_1"], row["tilt_2"], row["phi_12"], row["a_1"], row["a_2"],
        m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, F_REF, row["phase"],
    )  # fmt: skip
    fv = lal.CreateREAL8Vector(len(freqs))
    fv.data = freqs
    alpha, _, cosbeta = method(fv, m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, *spins, iota, F_REF, 2, lal.CreateDict())
    return np.asarray(alpha.data), np.arccos(np.clip(np.asarray(cosbeta.data), -1.0, 1.0))


def precession_stats(post: pd.DataFrame, band: dict) -> pd.DataFrame:
    freqs = np.unique(np.append(np.geomspace(band["f_lo"], band["f_hi"], NFREQ), band["f_star"]))
    istar = int(np.searchsorted(freqs, band["f_star"]))
    rows = []
    for _, row in post.iterrows():
        alpha, beta = angles(row, freqs, lalsimulation.SimIMRPhenomXPPNAngles)
        _, beta_msa = angles(row, freqs[[istar]], lalsimulation.SimIMRPhenomXPMSAAngles)
        q = row["mass_ratio"]
        a1, a2, t1, t2 = row["a_1"], row["a_2"], row["tilt_1"], row["tilt_2"]
        rows.append({
            "q": q,
            "chi_eff": (a1 * np.cos(t1) + q * a2 * np.cos(t2)) / (1 + q),
            "chi_p": max(a1 * np.sin(t1), (4 * q + 3) / (4 + 3 * q) * q * a2 * np.sin(t2)),
            "beta_star": np.degrees(beta[istar]),
            "beta_star_msa": np.degrees(beta_msa[0]),
            "n_prec": abs(alpha[-1] - alpha[0]) / (2 * np.pi),
            "log_likelihood": row["log_likelihood"],
        })  # fmt: skip
    return pd.DataFrame(rows)


def read_posterior(outdir: Path, label: str) -> pd.DataFrame:
    return bilby.result.read_in_result(str(outdir / label / f"{label}_result.hdf5")).posterior


def summarise(outdir: Path, study: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    summary, samples = [], []
    for index, inj in enumerate(STUDIES[study]):
        band = snr_band(inj)
        prec = read_posterior(outdir, f"inj{index}_precessing")
        aligned = read_posterior(outdir, f"inj{index}_aligned")
        pick = prec.iloc[rng.choice(len(prec), min(NSAMPLES, len(prec)), replace=False)]
        stats = precession_stats(pick, band)
        best = stats.loc[stats["log_likelihood"].idxmax()]
        row = {"index": index, "m1": inj["m1"], "dt_ms": inj["dt"] * 1e3, "flux": inj["flux"], **band}
        for name in ("beta_star", "beta_star_msa", "n_prec", "chi_eff", "q"):
            lo, mid, hi = np.percentile(stats[name], (5, 50, 95))
            row.update({f"{name}_lo": lo, f"{name}_med": mid, f"{name}_hi": hi})
        row.update({f"best_{k}": best[k] for k in ("beta_star", "n_prec", "q", "chi_eff", "chi_p")})
        row["corr_beta_chi_eff"] = stats[["beta_star", "chi_eff"]].corr(method="spearman").iloc[0, 1]
        row["corr_beta_q"] = stats[["beta_star", "q"]].corr(method="spearman").iloc[0, 1]
        row["maxlnl_prec"] = prec["log_likelihood"].max()
        row["maxlnl_aligned"] = aligned["log_likelihood"].max()
        summary.append(row)
        samples.append(stats.assign(index=index))
        print(f"inj{index} done", flush=True)
    return pd.DataFrame(summary), pd.concat(samples, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("results/pe_generality"))
    parser.add_argument("--study", choices=sorted(STUDIES), default="generality")
    args = parser.parse_args()
    summary, samples = summarise(args.outdir, args.study)
    summary.to_csv(args.outdir / "diagnostics.csv", index=False)
    samples.to_csv(args.outdir / "diagnostics_samples.csv", index=False)
    cols = ["index", "m1", "dt_ms", "flux", "f_lo", "f_star", "f_hi", "fringes_band", "fringe_phase",
            "beta_star_med", "beta_star_msa_med", "n_prec_med", "best_beta_star", "best_n_prec",
            "corr_beta_chi_eff", "corr_beta_q", "maxlnl_prec", "maxlnl_aligned"]  # fmt: skip
    print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    main()
