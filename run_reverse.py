import itertools
import time
import traceback

import numpy as np

from fit import PrecessingProblem, joint_reverse_rows
from run_grid import INJECTION_PSI, execute, fibonacci_sky, parse_args, task_rows

PRESETS = {
    "pilot": {
        "sources": [(100.0, 80.0), (40.0, 10.0)],
        "chi1": [0.4, 0.8],
        "cos_tilt1": [0.0],
        "chi2z": [0.0],
        "cos_theta_jn": [0.9, 0.5, 0.0],
        "phi_jl": [0.0],
        "nsky": 6,
        "networks": ["H", "HL", "HLV", "HLVI", "ET"],
        "sky_modes": ["free"],
    },
}
TASK_FIELDS = ["m1", "m2", "chi1", "cos_tilt1", "chi2z", "cos_theta_jn", "phi_jl", "isky", "sky_mode"]
RESULT_FIELDS = [
    "ra",
    "dec",
    "psi",
    "chi_perp",
    "snr_dref",
    "ff_aligned",
    "ff_lensed",
    "ff_aligned_own",
    "ff_lensed_own",
    "ff_lensed_de",
    "ff_lensed_multistart",
    "best_from",
    "rho_false_lensing",
    "rho_crit_lensing",
    "best_dt_ms",
    "best_flux_ratio",
    "best_n_fringes",
    "nfev",
    "seconds",
    "best_aligned",
    "best_lensed",
]


def make_tasks(preset: dict) -> list[dict]:
    sky = fibonacci_sky(preset["nsky"])
    return [
        {
            "m1": m1,
            "m2": m2,
            "chi1": chi1,
            "cos_tilt1": cos_tilt1,
            "chi2z": chi2z,
            "cos_theta_jn": cos_th,
            "phi_jl": phi_jl,
            "isky": isky,
            "sky_mode": mode,
            "networks": preset["networks"],
            "ra": sky[isky][0],
            "dec": sky[isky][1],
            "psi": INJECTION_PSI,
        }
        for (m1, m2), chi1, cos_tilt1, chi2z, cos_th, phi_jl, isky, mode in itertools.product(
            preset["sources"],
            preset["chi1"],
            preset["cos_tilt1"],
            preset["chi2z"],
            preset["cos_theta_jn"],
            preset["phi_jl"],
            range(len(sky)),
            preset["sky_modes"],
        )
    ]


def run_task(args: tuple) -> list[dict] | None:
    task, seed, popsize, maxiter = args
    t0 = time.perf_counter()
    try:
        problems = {
            net: PrecessingProblem(
                task["m1"],
                task["m2"],
                task["chi1"],
                task["cos_tilt1"],
                task["chi2z"],
                float(np.arccos(task["cos_theta_jn"])),
                task["phi_jl"],
                task["psi"],
                task["ra"],
                task["dec"],
                net,
                task["sky_mode"] == "free",
            )
            for net in task["networks"]
        }
        rows = joint_reverse_rows(problems, seed, popsize, maxiter)
    except Exception:  # noqa: BLE001
        print(f"task failed: {task}\n{traceback.format_exc()}", flush=True)
        return None
    return task_rows(task, rows, t0)


def describe(rows: list[dict]) -> str:
    r = rows[0]
    head = f"{r['m1']:g}+{r['m2']:g} chi1={r['chi1']} cos_thJN={r['cos_theta_jn']} sky={r['isky']}:"
    body = " ".join(
        f"{x['network']} {x['ff_aligned']:.4f}/{x['ff_lensed']:.4f}({x['best_from']})" for x in rows
    )
    return f"{head} {body}"


def main() -> None:
    args = parse_args(PRESETS)
    execute(make_tasks(PRESETS[args.preset]), TASK_FIELDS, RESULT_FIELDS, run_task, describe, args)


if __name__ == "__main__":
    main()
