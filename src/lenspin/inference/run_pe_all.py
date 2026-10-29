import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lenspin.inference.pe import STUDIES, STUDY_MODELS


def launch(job: tuple[int, str], study: str, outdir: Path, npool: int, nlive: int) -> str:
    index, model = job
    label = f"inj{index}_{model}"
    if (outdir / f"{label}.json").exists():
        return f"{label}: already done"
    with (outdir / f"{label}.log").open("w") as log:
        code = subprocess.call(
            [sys.executable, "-m", "lenspin.inference.pe", "--study", study, "--index", str(index), "--model", model,
             "--outdir", str(outdir), "--npool", str(npool), "--nlive", str(nlive)],
            stdout=log,
            stderr=subprocess.STDOUT,
        )  # fmt: skip
    return f"{label}: exit {code}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", choices=sorted(STUDIES), default="network")
    parser.add_argument("--outdir", type=Path, default=Path("results/pe"))
    parser.add_argument("--concurrent", type=int, default=7)
    parser.add_argument("--npool", type=int, default=16)
    parser.add_argument("--nlive", type=int, default=400)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    jobs = [(i, m) for m in STUDY_MODELS[args.study] for i in range(len(STUDIES[args.study]))]
    with ThreadPoolExecutor(args.concurrent) as pool:
        for message in pool.map(lambda j: launch(j, args.study, args.outdir, args.npool, args.nlive), jobs):
            print(message, flush=True)


if __name__ == "__main__":
    main()
