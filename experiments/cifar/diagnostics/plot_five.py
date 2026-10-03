import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/mpl-cifar"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--prefix", default="new-cifar-budget")
    args = parser.parse_args()
    plt.style.use(ROOT / "paper.mplstyle")
    cases = [
        ("jfb", "JFB", "#0072B2", "--"),
        ("two_terms", "Two terms", "#D55E00", "-."),
        ("three_terms", "Three terms", "#009E73", ":"),
        ("damped_two", r"Two terms, $\lambda=1/2$", "#CC79A7", (0, (5, 2, 1, 2))),
        ("phantom_baseline", "Phantom", "#111111", "-"),
    ]
    records = []
    for name, label, color, style in cases:
        result = json.loads((args.results / name / "result.json").read_text())
        history = json.loads((args.results / name / "history.json").read_text())
        assert result["status"] == "completed" and len(history) == 50, name
        records.append((label, color, style, history))
    for metric, ylabel, suffix in [
        ("online_train_loss", "Training cross-entropy", "loss"),
        (
            "max_batch_mean_forward_relative_residual",
            "Forward relative residual",
            "residual",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(5.1, 3.6))
        for label, color, style, history in records:
            ax.plot(
                [r["epoch"] for r in history],
                [r[metric] for r in history],
                label=label,
                color=color,
                linestyle=style,
            )
        ax.set(xlabel="Epoch", ylabel=ylabel, yscale="log", xlim=(1, 50))
        if suffix == "loss":
            ax.legend(loc="lower left")
        fig.tight_layout()
        for extension in ["pdf", "png"]:
            fig.savefig(ROOT / "images" / f"{args.prefix}-{suffix}.{extension}")
        plt.close(fig)


if __name__ == "__main__":
    main()
