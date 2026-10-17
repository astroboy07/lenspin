import argparse
import csv
import itertools
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from fit import Problem, run_problem

PRESETS = {
    "pilot": {
        "source_dts": [((100.0, 80.0), [5.0, 20.0])],
        "flux_ratios": [0.3, 0.9],
        "cosi": [0.9, 0.5, 0.0],
        "nsky": 6,
        "networks": ["H", "HL", "HLV", "HLVI", "ET"],
        "sky_modes": ["free"],
    },
    "trimmed": {
        "source_dts": [((30.0, 30.0), [3.0, 6.0, 12.0]), ((100.0, 80.0), [9.0, 19.0, 38.0])],
        "flux_ratios": [0.1, 0.3, 0.6, 0.9],
        "cosi": [0.9, 0.7, 0.5, 0.3, 0.0],
        "nsky": 8,
        "networks": ["HL", "HLV", "HLVI", "ET"],
        "sky_modes": ["free"],
    },
    "full": {
        "source_dts": [
            ((30.0, 30.0), [2.0, 5.0, 10.0, 20.0, 50.0]),
            ((100.0, 80.0), [2.0, 5.0, 10.0, 20.0, 50.0]),
        ],
        "flux_ratios": [0.1, 0.3, 0.6, 0.9],
        "cosi": [1.0, 0.9, 0.7, 0.5, 0.3, 0.0],
        "nsky": 12,
        "networks": ["HL", "HLV", "HLVI", "ET"],
        "sky_modes": ["fixed", "free"],
    },
}
INJECTION_PSI = 0.3
KEY_FIELDS = ["m1", "m2", "dt_ms", "flux_ratio", "cosi", "isky", "network", "sky_mode"]
RESULT_FIELDS = [
    "ra",
    "dec",
    "psi",
    "n_fringes",
    "snr_dref",
    "ff_aligned",
    "ff_precessing",
    "rho_crit_aligned",
    "rho_crit_precessing",
    "d_crit_mpc",
    "nfev",
    "seconds",
    "best_aligned",
    "best_precessing",
]


def fibonacci_sky(n: int) -> list[tuple[float, float]]:
    i = np.arange(n)
    dec = np.arcsin(1.0 - (2 * i + 1) / n)
    ra = (i * np.pi * (3.0 - np.sqrt(5.0))) % (2 * np.pi)
    return list(zip(ra.tolist(), dec.tolist()))


def make_tasks(preset: dict) -> list[dict]:
    sky = fibonacci_sky(preset["nsky"])
    tasks = []
    combos = (
        ((m1, m2), *rest)
        for (m1, m2), dts in preset["source_dts"]
        for rest in itertools.product(
            dts,
            preset["flux_ratios"],
            preset["cosi"],
            range(len(sky)),
            preset["networks"],
            preset["sky_modes"],
        )
    )
    for (m1, m2), dt, flux, cosi, isky, net, mode in combos:
        tasks.append(
            {
                "m1": m1,
                "m2": m2,
                "dt_ms": dt,
                "flux_ratio": flux,
                "cosi": cosi,
                "isky": isky,
                "network": net,
                "sky_mode": mode,
                "ra": sky[isky][0],
                "dec": sky[isky][1],
                "psi": INJECTION_PSI,
            },
        )
    return tasks


def task_key(row: dict) -> tuple:
    return tuple(str(float(row[k])) if k not in ("network", "sky_mode") else row[k] for k in KEY_FIELDS)


def run_task(args: tuple) -> dict:
    task, seed, popsize, maxiter = args
    t0 = time.perf_counter()
    problem = Problem(
        task["m1"],
        task["m2"],
        task["dt_ms"] * 1e-3,
        task["flux_ratio"],
        task["cosi"],
        task["psi"],
        task["ra"],
        task["dec"],
        task["network"],
        task["sky_mode"] == "free",
    )
    try:
        row = run_problem(problem, seed, popsize, maxiter)
    except Exception:  # noqa: BLE001
        print(f"task failed: {task}\n{traceback.format_exc()}", flush=True)
        return None
    return {**task, **row, "seconds": time.perf_counter() - t0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS), default="pilot")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--popsize", type=int, default=12)
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tasks = make_tasks(PRESETS[args.preset])
    done = set()
    if args.out.exists():
        with args.out.open() as fh:
            done = {task_key(row) for row in csv.DictReader(fh)}
    todo = [
        (t, args.seed + i, args.popsize, args.maxiter) for i, t in enumerate(tasks) if task_key(t) not in done
    ]
    todo = todo[: args.limit] if args.limit else todo
    print(f"{len(tasks)} tasks in preset {args.preset}, {len(done)} done, running {len(todo)}", flush=True)

    fields = KEY_FIELDS + RESULT_FIELDS
    new_file = not args.out.exists()
    t_start = time.perf_counter()
    with args.out.open("a", newline="") as fh, Pool(args.workers) as pool:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if new_file:
            writer.writeheader()
            fh.flush()
        for n, row in enumerate(pool.imap_unordered(run_task, todo), 1):
            if row is None:
                continue
            writer.writerow(row)
            fh.flush()
            elapsed = time.perf_counter() - t_start
            print(
                f"[{n}/{len(todo)}] {row['network']} {row['sky_mode']} dt={row['dt_ms']}ms "
                f"I={row['flux_ratio']} cosi={row['cosi']} sky={row['isky']}: "
                f"FF_al={row['ff_aligned']:.4f} FF_pr={row['ff_precessing']:.4f} "
                f"rho_crit={row['rho_crit_precessing']:.1f} ({row['seconds']:.0f} s; "
                f"eta {elapsed / n * (len(todo) - n) / 60:.0f} min)",
                flush=True,
            )


if __name__ == "__main__":
    main()
