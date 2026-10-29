import argparse
import time

import numpy as np

from lenspin.fitting.fit import PrecessingProblem, Problem, run_problem, run_reverse_problem

CONFIG = {"m1": 100.0, "m2": 80.0, "cosi": 0.5, "psi": 0.3, "ra": 1.0, "dec": 0.4}
LENS = {"dt": 0.01, "flux_ratio": 0.5}
PRECESSION = {"chi1": 0.8, "cos_tilt1": 0.0, "chi2z": 0.0, "theta_jn": 1.0, "phi_jl": 0.0}
TRUTH_TOL = 1e-5
ORDER_TOL = 1e-4


def problem(network: str, dt: float, flux_ratio: float, sky_free: bool = False) -> Problem:
    c = CONFIG
    return Problem(
        c["m1"], c["m2"], dt, flux_ratio, c["cosi"], c["psi"], c["ra"], c["dec"], network, sky_free
    )


def report(name: str, ok: bool, detail: str) -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--popsize", type=int, default=6)
    parser.add_argument("--maxiter", type=int, default=25)
    args = parser.parse_args()
    results = []

    for net in ["H", "HL", "HLV", "HLVI", "ET"]:
        p = problem(net, 0.0, 0.0)
        theta = np.arccos(CONFIG["cosi"])
        ff_al = p.ff_aligned([p.mchirp, p.q, 0.0, 0.0])
        ff_pr = p.ff_precessing([p.mchirp, p.q, 0.0, 1.0, 0.0, theta, 0.0])
        ok = min(ff_al, ff_pr) > 1.0 - TRUTH_TOL
        results.append(
            report(f"unlensed truth overlap {net}", ok, f"aligned {ff_al:.7f}, precessing {ff_pr:.7f}")
        )

    rows = {}
    for net in ["H", "HL", "HLVI"]:
        t0 = time.perf_counter()
        rows[net] = run_problem(problem(net, **LENS), 1, args.popsize, args.maxiter)
        r = rows[net]
        ok = r["ff_precessing"] >= r["ff_aligned"] - ORDER_TOL
        detail = (
            f"FF aligned {r['ff_aligned']:.5f}, precessing {r['ff_precessing']:.5f}, "
            f"rho_crit {r['rho_crit_precessing']:.1f}, {r['nfev']} evals in {time.perf_counter() - t0:.0f} s"
        )
        results.append(report(f"precessing >= aligned {net}", ok, detail))

    gap = abs(rows["HL"]["ff_precessing"] - rows["H"]["ff_precessing"])
    print(f"[INFO] H vs HL precessing FF differ by {gap:.2e} (expected small: H and L are nearly co-aligned)")

    seeds = [
        run_problem(problem("HL", **LENS), s, args.popsize, args.maxiter)["ff_precessing"] for s in (2, 3)
    ]
    seeds.append(rows["HL"]["ff_precessing"])
    spread = max(seeds) - min(seeds)
    print(f"[INFO] HL seed spread {spread:.2e} over 3 seeds (target < 1e-3 at production settings)")

    for net in ["HL", "HLVI"]:
        c, s = CONFIG, PRECESSION
        p = PrecessingProblem(
            c["m1"],
            c["m2"],
            0.0,
            1.0,
            0.0,
            s["theta_jn"],
            s["phi_jl"],
            c["psi"],
            c["ra"],
            c["dec"],
            net,
            False,
        )
        x = [p.mchirp, p.q, 0.0, 0.0]
        ff_al, ff_le = p.ff_aligned(x), p.ff_lensed([*x, 30.0, 0.0])
        ok = min(ff_al, ff_le) > 1.0 - TRUTH_TOL
        results.append(
            report(f"no-precession truth {net}", ok, f"aligned {ff_al:.7f}, lensed I=0 {ff_le:.7f}")
        )
        p = PrecessingProblem(c["m1"], c["m2"], *s.values(), c["psi"], c["ra"], c["dec"], net, False)
        ff_pr = p.ff_precessing(
            [p.mchirp, p.q, s["chi1"], s["cos_tilt1"], s["chi2z"], s["theta_jn"], s["phi_jl"]]
        )
        results.append(report(f"precessing injection truth {net}", ff_pr > 1.0 - TRUTH_TOL, f"{ff_pr:.7f}"))
        t0 = time.perf_counter()
        r = run_reverse_problem(p, 1, args.popsize, args.maxiter)
        ok = r["ff_lensed"] >= r["ff_aligned"] - ORDER_TOL
        detail = (
            f"FF aligned {r['ff_aligned']:.5f}, lensed {r['ff_lensed']:.5f} (dt {r['best_dt_ms']:.1f} ms, "
            f"I {r['best_flux_ratio']:.2f}), rho_false_lensing {r['rho_false_lensing']:.1f}, "
            f"{r['nfev']} evals in {time.perf_counter() - t0:.0f} s"
        )
        results.append(report(f"lensed >= aligned {net}", ok, detail))

    print("ALL PASS" if all(results) else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
