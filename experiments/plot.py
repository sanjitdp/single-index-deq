import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT
DESTINATION = ROOT / "images"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "mpl-alt"))
sys.path.insert(0, str(SOURCE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import json


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, SOURCE / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


original_subplots = plt.subplots
original_savefig = Figure.savefig
original_legend = Axes.legend
cifar_mode = False


def subplots(*args, **kwargs):
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.labelsize": 9.5,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "savefig.bbox": None,
        }
    )
    kwargs["figsize"] = (0.47 * 6, 3.0 if cifar_mode else 2.45)
    return original_subplots(*args, **kwargs)


def savefig(self, filename, *args, **kwargs):
    filename = Path(filename)
    if filename.suffix == ".png":
        return
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    triple = "sigmoid-far" in filename.name
    trajectory = "dynamics" in filename.name
    font = 9
    paired_trajectory = not triple and (trajectory or "distances-" in filename.name)
    panel_width = (
        (2.22 if trajectory else 1.86)
        if triple
        else (2.94 if paired_trajectory else 2.82)
    )
    self.set_size_inches(panel_width, 2.15 if triple else 2.45)
    ax = self.axes[0]

    if triple:

        left = (0.50 if "loss-" in filename.name else 0.40) / panel_width
        bottom, width, height = 0.23, 1.30 / panel_width, 0.70
    elif paired_trajectory:
        left, bottom, width, height = 0.19, 0.21, 0.63, 0.72
    else:
        left, bottom, width, height = 0.22, 0.21, 0.73, 0.72
    ax.set_position([left, bottom, width, height])
    ax.tick_params(labelsize=font, pad=2)
    ax.xaxis.label.set_size(font)
    ax.yaxis.label.set_size(font)
    ax.xaxis.labelpad = 5
    ax.yaxis.labelpad = 2 if triple else 4
    if "updates" in ax.get_xlabel().lower():
        if ax.get_xlim()[1] >= 1000:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x / 1000:g}"))
            ax.set_xlabel(r"Updates ($10^3$)", fontsize=font)
        else:
            ax.set_xlabel("Updates", fontsize=font)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3 if triple else 4))
    handles, labels = ax.get_legend_handles_labels()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    labels = [label.replace("JFB (1 term)", "JFB") for label in labels]
    labels = ["Damped two" if r"\lambda" in label else label for label in labels]
    labels = ["Zero-risk set" if r"\theta_1=2" in label else label for label in labels]
    labels = ["Trivial solution" if label == r"$(0,1)$" else label for label in labels]
    labels = ["Target" if label == r"Target $(2,0)$" else label for label in labels]
    external = filename.name.startswith(("new-approximate-", "new-cifar-"))
    if external:
        if filename.name in ("new-approximate-linear.pdf", "new-cifar-budget-loss.pdf"):
            stem = (
                "new-approximate-legend"
                if "approximate" in filename.name
                else "new-cifar-legend"
            )
            labels = [
                {"2 terms": "Two terms", "3 terms": "Three terms"}.get(s, s)
                for s in labels
            ]
            legend_fig = plt.figure(figsize=(6, 0.48))
            legend_fig.legend(
                handles,
                labels,
                loc="center",
                ncol=len(labels),
                fontsize=9,
                frameon=True,
                facecolor="white",
                edgecolor="0.8",
                borderpad=0.8,
                columnspacing=1.4,
                handlelength=2,
                handletextpad=0.6,
            )
            for destination in (DESTINATION,):
                original_savefig(
                    legend_fig, destination / f"{stem}.pdf", bbox_inches=None
                )
                original_savefig(
                    legend_fig, destination / f"{stem}.png", dpi=180, bbox_inches=None
                )
            plt.close(legend_fig)
    elif handles:
        location = "best"
        if trajectory:
            location = "upper center" if triple else "lower center"
            if "lm" in filename.name:
                location = "upper right"
                lo, hi = ax.get_ylim()
                ax.set_ylim(lo, hi + 0.35 * (hi - lo))
        elif filename.name.startswith("new-nonlinearity-"):
            location = "lower left"
        box = original_legend(
            ax,
            handles,
            labels,
            loc=location,
            fontsize=font,
            frameon=True,
            facecolor="white",
            edgecolor="0.8",
            framealpha=0.95,
            borderpad=0.5,
            labelspacing=0.45,
            handlelength=1.7,
            handletextpad=0.55,
        )
        if trajectory:
            for handle in box.get_lines():
                handle.set_linewidth(1.8)
                if handle.get_marker() == "*":
                    handle.set_markersize(6)
    if trajectory:
        cax = self.axes[1]
        cax.set_box_aspect(None)
        cax.set_aspect("auto")

        cax.set_position(
            [
                left + width + 0.17 / panel_width,
                bottom + 0.075 * height,
                0.075 / panel_width,
                0.85 * height,
            ]
        )
        cax.set_ylabel("")
        cax.set_title("Updates", fontsize=font - 1, pad=7)
        cax.tick_params(labelsize=font - 1, pad=3, length=2, width=0.5)
        end = ax.collections[-1].norm.vmax
        cax.set_yticks([0, end / 2, end])
        for spine in cax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_edgecolor("0.4")
        cax.yaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: f"{x / 1000:g}k" if x >= 1000 else f"{x:g}")
        )
    kwargs["bbox_inches"] = None
    for destination in (DESTINATION,):
        original_savefig(self, destination / filename.name, *args, **kwargs)
        original_savefig(
            self,
            destination / filename.with_suffix(".png").name,
            dpi=180,
            bbox_inches=None,
        )


def main():
    global cifar_mode
    DESTINATION.mkdir(exist_ok=True)
    plt.subplots = subplots
    Figure.savefig = savefig
    approximate = load("alt_approximate", "synthetic/approximate.py")
    approximate.plot([100, 101, 102, 103, 104])
    nonlinear = load("alt_nonlinearity", "synthetic/nonlinearity.py")
    config = json.loads((nonlinear.RESULTS / "config.json").read_text())
    epsilons = [0.5, 0.25, 0.125]
    trajectories = {epsilon: [] for epsilon in epsilons}
    for seed in config["seeds"]:
        for epsilon in epsilons:
            with np.load(
                nonlinear.RESULTS / f"seed-{seed}-eps-{epsilon:g}.npz"
            ) as data:
                trajectories[epsilon].append({key: data[key] for key in data.files})
    nonlinear.plot(trajectories, epsilons, DESTINATION)
    legacy = load("alt_legacy", "synthetic/legacy_figures.py")
    legacy.plot_all()
    cifar = load("alt_cifar", "cifar/diagnostics/plot_five.py")
    cifar_mode = True
    sys.argv = [__file__, "--results", str(SOURCE / "results/cifar")]
    cifar.main()
    print("Rendered paper figures in images/.")


if __name__ == "__main__":
    main()
