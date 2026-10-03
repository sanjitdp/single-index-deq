import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, choices=range(5), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads((here / "configs.json").read_text())
    case = config["configs"][args.index]
    command = [sys.executable, str(here / "train.py"), "--phase", "final"]
    values = {
        "dataset": config["dataset"],
        "seed": config["seed"],
        "epochs": config["epochs"],
        "batch-size": config["batch_size"],
        "lr": config["learning_rate"],
        "forward-iterations": config["forward_iterations"],
        "forward-tolerance": config["forward_tolerance"],
        "method": case["method"],
        "terms": case["terms"],
        "damping": case["damping"],
        "device": args.device,
        "data": args.data,
        "output": args.output / case["name"],
    }
    for key, value in values.items():
        command.extend(["--" + key, str(value)])
    for key in ["download", "resume"]:
        if getattr(args, key):
            command.append("--" + key)
    print(subprocess.list2cmdline(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
