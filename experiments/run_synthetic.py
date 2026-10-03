import argparse
import concurrent.futures
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def execute(script):
    env = os.environ.copy()
    for key in ["OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS"]:
        env[key] = "1"
    log = ROOT / "tmp" / (script + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        subprocess.run(
            [sys.executable, str(ROOT / "synthetic" / (script + ".py"))],
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print(f"Completed {script}; log: {log}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--overwrite-results", action="store_true")
    args = parser.parse_args()
    if not args.plot_only:
        if not args.overwrite_results and any((ROOT / "results").glob("*/*.npz")):
            parser.error(
                "Saved results exist. Use --overwrite-results to replace them, or --plot-only."
            )
        (ROOT / "images").mkdir(exist_ok=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(execute, ["approximate", "nonlinearity", "legacy_figures"]))
        execute("select_linear_illustration")
    subprocess.run([sys.executable, str(ROOT / "plot.py")], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
