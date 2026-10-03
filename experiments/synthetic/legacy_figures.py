from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "mpl-root"))
for name in ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

OUT = ROOT / "results/legacy"
STYLE = ROOT / "paper.mplstyle"


DISPLAY_STEPS = {
    "linear": 100,
    "linear-near-singular": 20,
    "sigmoid": 12000,
    "sigmoid-far": 8000,
}


def sigmoid_equilibrium(x, theta, warm=None):

    c, b = theta
    if b >= 4:
        raise RuntimeError("The sigmoid's globally unique-root guard was lost.")
    y = expit(c * x) if warm is None else warm.copy()
    lower, upper = np.zeros_like(x), np.ones_like(x)
    for _ in range(100):
        value = expit(c * x + b * y)
        r = y - value
        if np.max(np.abs(r)) <= 1e-13:
            return y, float(np.max(np.abs(r)))
        lower = np.where(r <= 0, y, lower)
        upper = np.where(r > 0, y, upper)
        proposal = y - r / (1 - b * value * (1 - value))
        y = np.where(
            (proposal >= lower) & (proposal <= upper), proposal, (lower + upper) / 2
        )
    raise RuntimeError("Sigmoid root solver failed its residual check.")


def sigmoid_gradient(x, theta, target, warm=None):
    y, residual = sigmoid_equilibrium(x, theta, warm)
    value = expit(theta[0] * x + theta[1] * y)
    derivative = value * (1 - value)
    weights = 2 * (y - target) * derivative / (1 - theta[1] * derivative)
    return np.array([weights @ x, weights @ y]) / len(x), y, residual


def run(kind, theta0, steps, eta, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, 1000)
    target = 2 * x if kind == "linear" else expit(2 * x)
    theta = np.asarray(theta0, dtype=float).copy()
    history = np.empty((steps + 1, 2))
    risk = np.empty(steps + 1)
    maximum_residual = 0.0
    warm = None
    start = time.perf_counter()
    for t in range(steps + 1):
        if kind == "linear":
            phi = theta[0] / (1 - theta[1])
            prediction = phi * x
            gradient = (
                2 * np.mean(x * x) * (phi - 2) / (1 - theta[1]) * np.array([1.0, phi])
            )
        else:
            gradient, prediction, residual = sigmoid_gradient(x, theta, target, warm)
            warm = prediction
            maximum_residual = max(maximum_residual, residual)
        history[t] = theta
        risk[t] = np.mean((prediction - target) ** 2)
        if t < steps:
            theta -= eta * gradient
        if not np.all(np.isfinite(theta)):
            raise RuntimeError(f"Nonfinite iterate at {t}.")
    return {
        "theta": history,
        "risk": risk,
        "x": x,
        "target": target,
        "eta": eta,
        "seed": seed,
        "steps": steps,
        "max_residual": maximum_residual,
        "runtime_seconds": time.perf_counter() - start,
    }


def validate():
    x = np.linspace(-2, 2, 31)
    theta = np.array([1.8, 0.2])
    target = expit(2 * x)
    gradient, y, residual = sigmoid_gradient(x, theta, target)
    roots = np.array(
        [
            brentq(lambda z: z - expit(theta[0] * u + theta[1] * z), 0, 1, xtol=1e-14)
            for u in x
        ]
    )
    finite_difference = []
    for j in range(2):
        d = np.zeros(2)
        d[j] = 1e-6
        plus = sigmoid_equilibrium(x, theta + d)[0]
        minus = sigmoid_equilibrium(x, theta - d)[0]
        finite_difference.append(
            (np.mean((plus - target) ** 2) - np.mean((minus - target) ** 2)) / 2e-6
        )
    error = float(np.max(np.abs(gradient - finite_difference)))
    root_error = float(np.max(np.abs(roots - y)))
    assert error < 1e-9 and root_error < 2e-12
    return {
        "finite_difference_max_error": error,
        "brent_max_difference": root_error,
        "residual": residual,
    }


def save(fig, name):
    fig.tight_layout(pad=0.5)
    for suffix in ("pdf", "png"):
        fig.savefig(ROOT / "images" / f"replot-{name}.{suffix}")
    plt.close(fig)


def line_plot(y, name, ylabel, logarithmic=False):
    fig, ax = plt.subplots(figsize=(4.2, 3.15))
    ax.plot(np.arange(len(y)), y, color="#1f77b4")
    if logarithmic:
        ax.set_yscale("log")
    ax.set_xlabel("Gradient-descent updates")
    ax.set_ylabel(ylabel)
    if len(y) > 10000:
        ax.ticklabel_format(axis="x", style="sci", scilimits=(3, 3), useMathText=True)
    save(fig, name)


def trajectory(theta, name, linear):
    fig, ax = plt.subplots(figsize=(4.2, 3.15))
    if linear:
        lo = min(0.0, float(theta[:, 0].min())) - 0.12
        hi = max(0.0, float(theta[:, 0].max())) + 0.12
        grid = np.linspace(lo, hi, 200)
        ax.plot(
            grid,
            1 - grid / 2,
            "--",
            color="#2ca02c",
            linewidth=1.1,
            label=r"$\theta_1=2(1-\theta_2)$",
        )
        ax.plot(0, 1, "r*", markersize=8, label=r"$(0,1)$")
        ax.set_xlim(lo, hi)
        ax.set_ylim(
            min(float(theta[:, 1].min()), 1) - 0.15,
            max(float(theta[:, 1].max()), 1) + 0.15,
        )
    else:
        ax.plot(2, 0, "r*", markersize=8, label=r"Target $(2,0)$")
    ax.plot(theta[:, 0], theta[:, 1], color="black", alpha=0.25, linewidth=0.7)

    indices = np.unique(
        np.r_[
            np.linspace(0, len(theta) - 1, min(1000, len(theta)), dtype=int),
            len(theta) - 1,
        ]
    )
    scatter = ax.scatter(
        theta[indices, 0],
        theta[indices, 1],
        c=indices,
        cmap="viridis",
        s=7,
        linewidths=0,
        rasterized=True,
    )
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.legend(loc="best", fontsize=9)
    bar = fig.colorbar(scatter, ax=ax, pad=0.025)
    bar.set_label("Updates", labelpad=5)
    bar.ax.tick_params(labelsize=10)
    save(fig, name)


def function_plot(theta, name, linear):
    grid = np.linspace(-2, 2, 250)
    true = 2 * grid if linear else expit(2 * grid)
    pred = (
        theta[0] / (1 - theta[1]) * grid
        if linear
        else sigmoid_equilibrium(grid, theta)[0]
    )
    fig, ax = plt.subplots(figsize=(4.2, 3.15))
    ax.plot(grid, true, color="blue", label="Target")
    ax.plot(grid, pred, "--", color="red", label="Learned")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.legend(loc="best")
    save(fig, name)


def plot_all():
    plt.style.use(STYLE)
    for key, linear, far in (
        ("linear", True, False),
        ("linear-near-singular", True, False),
        ("sigmoid", False, False),
        ("sigmoid-far", False, True),
    ):

        data_key = key + "-extended" if not linear else key
        if key == "linear-near-singular":
            data_key = "linear-near-singular-illustration"
        data = np.load(OUT / f"{data_key}.npz")
        stop = DISPLAY_STEPS[key] + 1
        theta, risk = data["theta"][:stop], data["risk"][:stop]
        if key == "linear-near-singular":
            suffix = "lm-unstable"
        else:
            suffix = "lm" if linear else key
            line_plot(risk, f"loss-{suffix}", "Training risk")
            if not far:
                function_plot(data["theta"][-1], f"learned-{suffix}", linear)
        trajectory(theta, f"dynamics-{suffix}", linear)
        target = np.array([0.0, 1.0]) if linear else np.array([2.0, 0.0])
        line_plot(
            np.linalg.norm(theta - target, axis=1),
            f"distances-{suffix}",
            r"$\|\theta-(0,1)\|_2$" if linear else r"$\|\theta-(2,0)\|_2$",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:

        perturbation = np.random.default_rng(43).normal(0, 0.1, 2)
        protocols = [
            ("linear", "linear", perturbation, 200, 0.01),
            (
                "linear-near-singular",
                "linear",
                perturbation + np.array([0, 1]),
                200,
                0.01,
            ),
            ("sigmoid", "sigmoid", perturbation, 4000, 0.1),
            ("sigmoid-far", "sigmoid", perturbation + np.array([20, 0]), 4000, 2.0),
            ("sigmoid-extended", "sigmoid", perturbation, 40000, 0.1),
            (
                "sigmoid-far-extended",
                "sigmoid",
                perturbation + np.array([20, 0]),
                40000,
                2.0,
            ),
        ]
        summary = {
            "validation": validate(),
            "initialization_seed": 43,
            "data_seed": 42,
            "n": 1000,
            "distribution": "Uniform[0,1]",
            "note": "Fresh runs of manuscript protocols; original image trajectories were not saved. Both 4000-step sigmoid reruns are retained; extended runs use the identical data/initialization and 40000 steps after 4000 proved insufficient.",
            "runs": {},
        }
        for key, kind, initial, steps, eta in protocols:
            data = run(kind, initial, steps, eta)
            np.savez_compressed(OUT / f"{key}.npz", **data)
            row = {
                "steps": steps,
                "eta": eta,
                "initial_theta": initial.tolist(),
                "final_theta": data["theta"][-1].tolist(),
                "final_train_risk": float(data["risk"][-1]),
                "initial_train_risk": float(data["risk"][0]),
                "min_distance_singular": float(
                    np.min(np.linalg.norm(data["theta"] - [0, 1], axis=1))
                ),
                "max_residual": data["max_residual"],
                "runtime_seconds": data["runtime_seconds"],
            }
            summary["runs"][key] = row
            print(key, json.dumps(row), flush=True)
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_all()


if __name__ == "__main__":
    main()
