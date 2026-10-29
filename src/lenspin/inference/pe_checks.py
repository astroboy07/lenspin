import time

import lal
import numpy as np

from lenspin.fitting.fit import Problem
from lenspin.inference.pe import (
    INJECTIONS,
    MODELS,
    STUDIES,
    fiducial,
    injection_distance,
    injection_parameters,
    interferometer,
    make_ifos,
    make_likelihood,
    make_priors,
    segment_duration,
)
from lenspin.models.network import GPS_TIME
from lenspin.models.signals import D_REF_MPC

GEOMETRY_TOL = 1e-6
SNR_TOL = 1e-3
RB_TOL = 0.05
PERTURBATIONS = 6


def check_geometry() -> bool:
    rng = np.random.default_rng(0)
    gps = lal.LIGOTimeGPS(GPS_TIME)
    gmst = lal.GreenwichMeanSiderealTime(gps)
    worst = 0.0
    for prefix in ("H1", "L1", "V1", "I1"):
        ifo = interferometer(prefix, 20.0)
        det = next(d for d in lal.CachedDetectors if d.frDetector.prefix == prefix)
        for _ in range(5):
            ra, dec, psi = rng.uniform(0, 2 * np.pi), np.arcsin(rng.uniform(-1, 1)), rng.uniform(0, np.pi)
            fp, fc = lal.ComputeDetAMResponse(det.response, ra, dec, psi, gmst)
            bp = ifo.antenna_response(ra, dec, GPS_TIME, psi, "plus")
            bc = ifo.antenna_response(ra, dec, GPS_TIME, psi, "cross")
            dt_lal = lal.TimeDelayFromEarthCenter(det.location, ra, dec, gps)
            dt_bilby = ifo.time_delay_from_geocenter(ra, dec, GPS_TIME)
            worst = max(worst, abs(fp - bp), abs(fc - bc), abs(dt_lal - dt_bilby))
    ok = worst < GEOMETRY_TOL
    print(f"[{'PASS' if ok else 'FAIL'}] bilby vs LAL antenna/time delay, worst diff {worst:.2e}", flush=True)
    return ok


def check_snr(inj: dict, name: str) -> bool:
    distance = injection_distance(inj)
    params = injection_parameters(inj, distance)
    ifos = make_ifos(inj, distance)
    snr_bilby = float(np.sqrt(sum(ifo.meta_data["optimal_SNR"] ** 2 for ifo in ifos)))
    p = Problem(
        inj["m1"],
        inj["m2"],
        inj["dt"],
        inj["flux"],
        inj["cosi"],
        params["psi"],
        params["ra"],
        params["dec"],
        inj["network"],
        False,
    )
    snr_ff = float(np.sqrt(p.ss)) * D_REF_MPC / distance
    ok = abs(snr_bilby / snr_ff - 1) < SNR_TOL
    target = f", target {inj['target_snr']:g}" if "target_snr" in inj else ""
    print(
        f"[{'PASS' if ok else 'FAIL'}] {name}: SNR bilby {snr_bilby:.3f} vs lenspin {snr_ff:.3f}{target}; "
        f"D {distance:.0f} Mpc, segment {segment_duration(inj):g} s",
        flush=True,
    )
    return ok


def check_relative_binning(index: int, update: bool) -> bool:
    inj = INJECTIONS[index]
    ifos = make_ifos(inj)
    rng = np.random.default_rng(1)
    ok_all = True
    for model in MODELS:
        priors = make_priors(model, inj)
        t0 = time.perf_counter()
        rb = make_likelihood(model, inj, ifos, priors, relative_binning=True)
        if not update:
            rb.update_fiducial_parameters = False
        setup = time.perf_counter() - t0
        exact = make_likelihood(model, inj, ifos, priors, relative_binning=False)
        base = fiducial(model, inj)
        diffs = []
        for k in range(PERTURBATIONS):
            point = dict(base)
            if k:
                point["chirp_mass"] *= 1 + rng.normal(0, 0.005)
                point["mass_ratio"] = float(np.clip(point["mass_ratio"] * (1 + rng.normal(0, 0.05)), 0.06, 1))
                point["theta_jn"] = float(np.clip(point["theta_jn"] + rng.normal(0, 0.1), 0, np.pi))
                point["phase"] += rng.normal(0, 0.2)
                point["geocent_time"] += rng.normal(0, 5e-4)
            for lk in (rb, exact):
                lk.parameters.update(point)
            t1 = time.perf_counter()
            l_rb = rb.log_likelihood_ratio()
            t_rb = time.perf_counter() - t1
            t1 = time.perf_counter()
            l_ex = exact.log_likelihood_ratio()
            t_ex = time.perf_counter() - t1
            diffs.append(abs(l_rb - l_ex))
        ok = max(diffs) < RB_TOL
        ok_all &= ok
        print(
            f"[{'PASS' if ok else 'FAIL'}] relative binning {model}: max |lnL_RB - lnL| {max(diffs):.3f} over "
            f"{PERTURBATIONS} points; setup {setup:.0f} s, eval {1e3 * t_rb:.1f} ms vs exact {1e3 * t_ex:.1f} ms",
            flush=True,
        )
    return ok_all


def main() -> None:
    results = [check_geometry(), check_snr(INJECTIONS[0], "network 0"), check_snr(INJECTIONS[3], "network 3")]
    results += [check_snr(inj, f"generality {i}") for i, inj in enumerate(STUDIES["generality"])]
    print("ALL PASS" if all(results) else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
