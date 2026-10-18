import argparse
import json
from pathlib import Path

import bilby
import lal
import numpy as np
from bilby.gw.source import lal_binary_black_hole, lal_binary_black_hole_relative_binning

from network import F_MIN_DEFAULT, GPS_TIME, NETWORKS, PSD_MODELS
from run_grid import INJECTION_PSI, fibonacci_sky
from signals import F_REF, chirp_mass, lens_factor

DURATION = 4.0
SAMPLING_FREQUENCY = 2048.0
APPROXIMANT = "IMRPhenomXP"
DISTANCE_MPC = 5300.0
MASSES = (100.0, 80.0)
LENS = {"lens_dt": 0.02, "lens_flux_ratio": 0.3}
NSKY = 6
INJECTIONS = [
    {"cosi": cosi, "isky": isky, "network": net}
    for cosi in (0.9, 0.5)
    for isky in (3, 0)
    for net in ("HL", "HLVI")
]
MODELS = ("aligned", "precessing", "lensed")
Q_MIN = 0.05
MCHIRP_SPAN = (0.7, 1.3)
SPIN_MAX = 0.99
DT_MAX = 0.06
TIME_WINDOW = 0.1
DISTANCE_RANGE = (300.0, 15000.0)
RB_EPSILON = 0.025
PSD_FLOOR = 1.0
SAMPLER = {"sampler": "dynesty", "nlive": 400, "sample": "acceptance-walk", "naccept": 60, "dlogz": 0.1}


def lensed_bbh(frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1, phi_12, a_2, tilt_2,
               phi_jl, theta_jn, phase, lens_dt, lens_flux_ratio, **kwargs):  # fmt: skip
    h = lal_binary_black_hole(
        frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl,
        theta_jn, phase, **kwargs,
    )  # fmt: skip
    if h is None:
        return None
    factor = lens_factor(frequency_array, lens_dt, lens_flux_ratio)
    return {k: v * factor for k, v in h.items()}


def lensed_bbh_relative_binning(frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1, phi_12,
                                a_2, tilt_2, phi_jl, theta_jn, phase, lens_dt, lens_flux_ratio,
                                **kwargs):  # fmt: skip
    freqs = frequency_array if kwargs.get("fiducial", 0) == 1 else np.asarray(kwargs["frequency_bin_edges"])
    h = lal_binary_black_hole_relative_binning(
        frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl,
        theta_jn, phase, **kwargs,
    )  # fmt: skip
    if h is None:
        return None
    factor = lens_factor(freqs, lens_dt, lens_flux_ratio)
    return {k: v * factor for k, v in h.items()}


def waveform_arguments() -> dict:
    return {
        "waveform_approximant": APPROXIMANT,
        "reference_frequency": F_REF,
        "minimum_frequency": F_MIN_DEFAULT,
    }


def interferometer(prefix: str, f_min: float) -> bilby.gw.detector.Interferometer:
    det = next(d for d in lal.CachedDetectors if d.frDetector.prefix == prefix)
    fr = det.frDetector
    freqs = np.arange(0.0, SAMPLING_FREQUENCY / 2 + 1.0 / DURATION, 1.0 / DURATION)
    series = lal.CreateREAL8FrequencySeries(
        "psd", lal.LIGOTimeGPS(0), 0.0, 1.0 / DURATION, lal.DimensionlessUnit, len(freqs) + 1
    )
    PSD_MODELS[prefix](series, f_min)
    psd = series.data.data[: len(freqs)]
    psd = np.where(psd > 0, psd, PSD_FLOOR)
    # LAL azimuths are clockwise from North; bilby's are counter-clockwise from East.
    return bilby.gw.detector.Interferometer(
        name=prefix,
        power_spectral_density=bilby.gw.detector.PowerSpectralDensity(frequency_array=freqs, psd_array=psd),
        minimum_frequency=f_min,
        maximum_frequency=SAMPLING_FREQUENCY / 2,
        length=4.0,
        latitude=np.degrees(fr.vertexLatitudeRadians),
        longitude=np.degrees(fr.vertexLongitudeRadians),
        elevation=fr.vertexElevation,
        xarm_azimuth=90.0 - np.degrees(fr.xArmAzimuthRadians),
        yarm_azimuth=90.0 - np.degrees(fr.yArmAzimuthRadians),
        xarm_tilt=fr.xArmAltitudeRadians,
        yarm_tilt=fr.yArmAltitudeRadians,
    )


def injection_parameters(inj: dict) -> dict:
    ra, dec = fibonacci_sky(NSKY)[inj["isky"]]
    m1, m2 = MASSES
    return {
        "chirp_mass": chirp_mass(m1, m2),
        "mass_ratio": m2 / m1,
        "a_1": 0.0,
        "a_2": 0.0,
        "tilt_1": 0.0,
        "tilt_2": 0.0,
        "phi_12": 0.0,
        "phi_jl": 0.0,
        "luminosity_distance": DISTANCE_MPC,
        "theta_jn": float(np.arccos(inj["cosi"])),
        "psi": INJECTION_PSI,
        "phase": 0.0,
        "ra": ra,
        "dec": dec,
        "geocent_time": float(GPS_TIME),
        **LENS,
    }


def make_ifos(inj: dict) -> bilby.gw.detector.InterferometerList:
    params = injection_parameters(inj)
    ifos = bilby.gw.detector.InterferometerList(
        [interferometer(p, F_MIN_DEFAULT) for p in NETWORKS[inj["network"]]]
    )
    ifos.set_strain_data_from_zero_noise(
        sampling_frequency=SAMPLING_FREQUENCY,
        duration=DURATION,
        start_time=params["geocent_time"] + 2.0 - DURATION,
    )
    wfg = bilby.gw.WaveformGenerator(
        duration=DURATION,
        sampling_frequency=SAMPLING_FREQUENCY,
        frequency_domain_source_model=lensed_bbh,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments(),
    )
    ifos.inject_signal(waveform_generator=wfg, parameters=params, raise_error=False)
    return ifos


def make_priors(model: str, inj: dict) -> bilby.gw.prior.BBHPriorDict:
    params = injection_parameters(inj)
    mc = params["chirp_mass"]
    t0 = params["geocent_time"]
    priors = bilby.gw.prior.BBHPriorDict(aligned_spin=model != "precessing")
    priors["chirp_mass"] = bilby.gw.prior.UniformInComponentsChirpMass(
        MCHIRP_SPAN[0] * mc, MCHIRP_SPAN[1] * mc, name="chirp_mass"
    )
    priors["mass_ratio"] = bilby.gw.prior.UniformInComponentsMassRatio(Q_MIN, 1.0, name="mass_ratio")
    priors["luminosity_distance"] = bilby.core.prior.PowerLaw(
        2.0, DISTANCE_RANGE[0], DISTANCE_RANGE[1], name="luminosity_distance"
    )
    priors["geocent_time"] = bilby.core.prior.Uniform(t0 - TIME_WINDOW, t0 + TIME_WINDOW, name="geocent_time")
    for key in ("mass_1", "mass_2"):
        priors.pop(key, None)
    if model == "precessing":
        priors["a_1"] = bilby.core.prior.Uniform(0.0, SPIN_MAX, name="a_1")
        priors["a_2"] = bilby.core.prior.Uniform(0.0, SPIN_MAX, name="a_2")
    else:
        for i in (1, 2):
            priors[f"chi_{i}"] = bilby.gw.prior.AlignedSpin(
                a_prior=bilby.core.prior.Uniform(0.0, SPIN_MAX), name=f"chi_{i}"
            )
    if model == "lensed":
        priors["lens_dt"] = bilby.core.prior.Uniform(0.0, DT_MAX, name="lens_dt")
        priors["lens_flux_ratio"] = bilby.core.prior.Uniform(0.0, 1.0, name="lens_flux_ratio")
    return priors


def fiducial(model: str, inj: dict) -> dict:
    params = injection_parameters(inj)
    keep = ["chirp_mass", "mass_ratio", "luminosity_distance", "theta_jn", "psi", "phase", "ra", "dec",
            "geocent_time"]  # fmt: skip
    fid = {k: params[k] for k in keep}
    if model == "precessing":
        fid.update(a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0)
    else:
        fid.update(chi_1=0.0, chi_2=0.0)
    if model == "lensed":
        fid.update(LENS)
    return fid


def make_likelihood(model: str, inj: dict, ifos, priors, relative_binning: bool = False, lookup=None):
    source = lensed_bbh_relative_binning if model == "lensed" else lal_binary_black_hole_relative_binning
    if not relative_binning:
        source = lensed_bbh if model == "lensed" else lal_binary_black_hole
    wfg = bilby.gw.WaveformGenerator(
        duration=DURATION,
        sampling_frequency=SAMPLING_FREQUENCY,
        frequency_domain_source_model=source,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments(),
    )
    if not relative_binning:
        if lookup is None:
            return bilby.gw.likelihood.GravitationalWaveTransient(ifos, wfg, priors=priors)
        return bilby.gw.likelihood.GravitationalWaveTransient(
            ifos,
            wfg,
            priors=priors,
            distance_marginalization=True,
            time_marginalization=True,
            distance_marginalization_lookup_table=str(lookup),
        )
    return bilby.gw.likelihood.RelativeBinningGravitationalWaveTransient(
        ifos,
        wfg,
        fiducial_parameters=fiducial(model, inj),
        update_fiducial_parameters=True,
        priors=priors,
        epsilon=RB_EPSILON,
    )


def run(index: int, model: str, outdir: Path, npool: int, nlive: int) -> dict:
    inj = INJECTIONS[index]
    label = f"inj{index}_{model}"
    ifos = make_ifos(inj)
    priors = make_priors(model, inj)
    (outdir / label).mkdir(parents=True, exist_ok=True)
    lookup = outdir / label / "distance_lookup.npz"
    likelihood = make_likelihood(model, inj, ifos, priors, lookup=lookup)
    sampler = {**SAMPLER, "nlive": nlive}
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        outdir=str(outdir / label),
        label=label,
        npool=npool,
        injection_parameters=injection_parameters(inj),
        save="hdf5",
        **sampler,
    )
    summary = {
        "index": index,
        **inj,
        "model": model,
        "log_evidence": float(result.log_evidence),
        "log_evidence_err": float(result.log_evidence_err),
        "log_noise_evidence": float(result.log_noise_evidence),
        "max_log_likelihood": float(np.max(result.posterior["log_likelihood"])),
        "network_snr": float(np.sqrt(sum(ifo.meta_data["optimal_SNR"] ** 2 for ifo in ifos))),
        "nlive": nlive,
    }
    (outdir / f"{label}.json").write_text(json.dumps(summary, indent=1))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("results/pe"))
    parser.add_argument("--npool", type=int, default=16)
    parser.add_argument("--nlive", type=int, default=SAMPLER["nlive"])
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    print(json.dumps(run(args.index, args.model, args.outdir, args.npool, args.nlive)))


if __name__ == "__main__":
    main()
