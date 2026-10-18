import argparse
import csv
import itertools
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from fit import Problem, joint_forward_rows

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
TASK_FIELDS = ["m1", "m2", "dt_ms", "flux_ratio", "cosi", "isky", "sky_mode"]
TEXT_FIELDS = ("network", "sky_mode")
RESULT_FIELDS = [
    "ra",
    "dec",
    "psi",
    "n_fringes",
    "snr_dref",
    "ff_aligned",
    "ff_precessing",
    "ff_aligned_own",
    "ff_precessing_own",
    "ff_precessing_de",
    "ff_precessing_multistart",
    "best_from",
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
    combos = (
        ((m1, m2), *rest)
        for (m1, m2), dts in preset["source_dts"]
        for rest in itertools.product(
            dts,
            preset["flux_ratios"],
            preset["cosi"],
            range(len(sky)),
            preset["sky_modes"],
        )
    )
    return [
        {
            "m1": m1,
            "m2": m2,
            "dt_ms": dt,
            "flux_ratio": flux,
            "cosi": cosi,
            "isky": isky,
            "sky_mode": mode,
            "networks": preset["networks"],
            "ra": sky[isky][0],
            "dec": sky[isky][1],
            "psi": INJECTION_PSI,
        }
        for (m1, m2), dt, flux, cosi, isky, mode in combos
    ]


def task_key(row: dict, key_fields: list[str]) -> tuple:
    return tuple(row[k] if k in TEXT_FIELDS else str(float(row[k])) for k in key_fields)


def task_rows(task: dict, rows: dict, t0: float) -> list[dict]:
    seconds = time.perf_counter() - t0
    base = {k: v for k, v in task.items() if k != "networks"}
    return [{**base, "network": net, **row, "seconds": seconds} for net, row in rows.items()]


def run_task(args: tuple) -> list[dict] | None:
    task, seed, popsize, maxiter = args
    t0 = time.perf_counter()
    try:
        problems = {
            net: Problem(
                task["m1"],
                task["m2"],
                task["dt_ms"] * 1e-3,
                task["flux_ratio"],
                task["cosi"],
                task["psi"],
                task["ra"],
                task["dec"],
                net,
                task["sky_mode"] == "free",
            )
            for net in task["networks"]
        }
        rows = joint_forward_rows(problems, seed, popsize, maxiter)
    except Exception:  # noqa: BLE001
        print(f"task failed: {task}\n{traceback.format_exc()}", flush=True)
        return None
    return task_rows(task, rows, t0)


def parse_args(presets: dict) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(presets), default="pilot")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--popsize", type=int, default=12)
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def execute(tasks, task_fields, result_fields, worker, describe, args) -> None:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        with args.out.open() as fh:
            done = {task_key(row, task_fields) for row in csv.DictReader(fh)}
    todo = [
        (t, args.seed + i, args.popsize, args.maxiter)
        for i, t in enumerate(tasks)
        if task_key(t, task_fields) not in done
    ]
    todo = todo[: args.limit] if args.limit else todo
    print(f"{len(tasks)} tasks in preset {args.preset}, {len(done)} done, running {len(todo)}", flush=True)

    fields = [*task_fields[:-1], "network", task_fields[-1], *result_fields]
    new_file = not args.out.exists()
    t_start = time.perf_counter()
    with args.out.open("a", newline="") as fh, Pool(args.workers) as pool:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if new_file:
            writer.writeheader()
            fh.flush()
        for n, rows in enumerate(pool.imap_unordered(worker, todo), 1):
            if rows is None:
                continue
            writer.writerows(rows)
            fh.flush()
            eta = (time.perf_counter() - t_start) / n * (len(todo) - n) / 60
            print(
                f"[{n}/{len(todo)}] {describe(rows)} ({rows[0]['seconds']:.0f} s; eta {eta:.0f} min)",
                flush=True,
            )


def describe(rows: list[dict]) -> str:
    r = rows[0]
    head = f"dt={r['dt_ms']}ms I={r['flux_ratio']} cosi={r['cosi']} sky={r['isky']} {r['sky_mode']}:"
    body = " ".join(
        f"{x['network']} {x['ff_aligned']:.4f}/{x['ff_precessing']:.4f}({x['best_from']})" for x in rows
    )
    return f"{head} {body}"


def main() -> None:
    args = parse_args(PRESETS)
    execute(make_tasks(PRESETS[args.preset]), TASK_FIELDS, RESULT_FIELDS, run_task, describe, args)


if __name__ == "__main__":
    main()
