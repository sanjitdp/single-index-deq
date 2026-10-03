from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "nonlinearity"
os.environ.setdefault("MPLCONFIGDIR", str(RESULTS / "matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import brentq


def equilibrium(x, theta, epsilon, tolerance=1e-12):

    u, v = x @ theta[:-1], theta[-1]
    if abs(v) * (1 + epsilon) >= 0.95:
        raise RuntimeError("Iterate left the prescribed contraction neighborhood")
    z = u / (1 - v)
    for count in range(1, 21):
        t = np.tanh(z)
        y = z + epsilon * t
        d = 1 + epsilon * (1 - t * t)
        residual = z - u - v * y
        if np.max(np.abs(residual)) <= tolerance:
            break
        z -= residual / (1 - v * d)
    else:
        raise RuntimeError("Newton solver did not converge")
    actual_residual = np.max(np.abs(y - (u + v * y + epsilon * np.tanh(u + v * y))))
    if actual_residual > 2 * tolerance * (1 + epsilon):
        raise RuntimeError(
            f"Equilibrium residual {actual_residual:g} exceeds tolerance"
        )
    return y, d, float(actual_residual), count


def loss_gradient(x, target, theta, epsilon):
    y, d, residual, count = equilibrium(x, theta, epsilon)
    error = y - target
    adjoint = 2 * error * d / (1 - theta[-1] * d)
    gradient = np.r_[x.T @ adjoint, y @ adjoint] / len(x)
    return float(np.mean(error * error)), gradient, residual, count


def make_data(seed, dimension, n_train, n_eval):
    rng = np.random.default_rng(seed)
    xi1 = rng.normal(size=dimension)
    xi1 /= np.linalg.norm(xi1)
    x_train = rng.normal(size=(n_train, dimension))
    x_eval = rng.normal(size=(n_eval, dimension))

    transverse = rng.normal(size=dimension)
    transverse -= (transverse @ xi1) * xi1
    transverse /= np.linalg.norm(transverse)
    xi = np.r_[xi1, 0.0]
    theta0 = xi + 0.1 * np.r_[-xi1, 1.0] / np.sqrt(2)
    theta0 += 0.03 * np.r_[transverse, 0.0]
    return x_train, x_eval, xi, theta0


def validate(x, xi, theta, epsilon):
    target = x @ xi[:-1] + epsilon * np.tanh(x @ xi[:-1])
    _, gradient, residual, _ = loss_gradient(x, target, theta, epsilon)
    fd = np.zeros_like(theta)
    step = 1e-6
    for j in range(len(theta)):
        offset = np.zeros_like(theta)
        offset[j] = step
        plus = loss_gradient(x, target, theta + offset, epsilon)[0]
        minus = loss_gradient(x, target, theta - offset, epsilon)[0]
        fd[j] = (plus - minus) / (2 * step)
    gradient_relative_error = np.linalg.norm(fd - gradient) / np.linalg.norm(gradient)
    assert gradient_relative_error < 1e-7, gradient_relative_error
    y, _, _, _ = equilibrium(x[:32], theta, epsilon)
    discrepancy = []
    for row, root in zip(x[:32], y):
        u, v = row @ theta[:-1], theta[-1]
        radius = (abs(u) + epsilon) / (1 - abs(v)) + 1
        scalar = brentq(
            lambda yy: yy - u - v * yy - epsilon * np.tanh(u + v * yy),
            -radius,
            radius,
            xtol=1e-14,
            rtol=1e-14,
        )
        discrepancy.append(abs(root - scalar))
    assert max(discrepancy) < 1e-10
    z = x @ xi[:-1]
    derivative = 1 + epsilon * (1 - np.tanh(z) ** 2)
    jacobian = derivative[:, None] * np.column_stack([x, target])
    hessian_eigenvalues = np.linalg.eigvalsh(2 * jacobian.T @ jacobian / len(x))
    gram = np.column_stack([x, np.tanh(z)])
    gram_eigenvalue = np.linalg.eigvalsh(gram.T @ gram / len(x))[0]
    assert gram_eigenvalue > 0
    linear_jacobian = np.column_stack([x, z])
    null_vector = np.r_[-xi[:-1], 1.0]
    assert np.linalg.norm(linear_jacobian @ null_vector) < 1e-10
    return {
        "gradient_finite_difference_relative_error": float(gradient_relative_error),
        "brentq_maximum_discrepancy": float(max(discrepancy)),
        "initial_equilibrium_residual": residual,
        "gram_minimum_eigenvalue": float(gram_eigenvalue),
        "hessian_minimum_eigenvalue": float(hessian_eigenvalues[0]),
        "hessian_maximum_eigenvalue": float(hessian_eigenvalues[-1]),
        "hessian_minimum_over_epsilon_squared": float(
            hessian_eigenvalues[0] / epsilon**2
        ),
    }


def run_one(x_train, x_eval, xi, theta0, epsilon, updates, step, interval):
    clock_start = time.perf_counter()
    theta = theta0.copy()
    train_index = x_train @ xi[:-1]
    eval_index = x_eval @ xi[:-1]
    target_train = train_index + epsilon * np.tanh(train_index)
    target_eval = eval_index + epsilon * np.tanh(eval_index)
    validation = validate(x_train, xi, theta0, epsilon)
    recorded = {
        name: []
        for name in [
            "updates",
            "theta",
            "parameter_error",
            "train_risk",
            "eval_risk",
            "wall_seconds",
            "residual",
        ]
    }
    maximum_residual = 0.0
    maximum_contraction = 0.0
    maximum_newton_steps = 0
    maximum_risk_increase = 0.0
    previous_risk = np.inf
    for iteration in range(updates + 1):
        risk, gradient, residual, newton_steps = loss_gradient(
            x_train, target_train, theta, epsilon
        )
        maximum_residual = max(maximum_residual, residual)
        maximum_newton_steps = max(maximum_newton_steps, newton_steps)
        maximum_contraction = max(maximum_contraction, abs(theta[-1]) * (1 + epsilon))
        maximum_risk_increase = max(maximum_risk_increase, risk - previous_risk)
        previous_risk = risk
        if iteration % interval == 0 or iteration == updates:
            eval_y, _, eval_residual, _ = equilibrium(x_eval, theta, epsilon)
            maximum_residual = max(maximum_residual, eval_residual)
            recorded["updates"].append(iteration)
            recorded["theta"].append(theta.copy())
            recorded["parameter_error"].append(np.linalg.norm(theta - xi))
            recorded["train_risk"].append(risk)
            recorded["eval_risk"].append(np.mean((eval_y - target_eval) ** 2))
            recorded["wall_seconds"].append(time.perf_counter() - clock_start)
            recorded["residual"].append(max(residual, eval_residual))
        if iteration != updates:
            theta -= step * gradient
    assert maximum_risk_increase < 1e-14, maximum_risk_increase
    statistics = {
        "epsilon": epsilon,
        "final_parameter_error": float(recorded["parameter_error"][-1]),
        "final_training_risk": float(recorded["train_risk"][-1]),
        "final_evaluation_risk": float(recorded["eval_risk"][-1]),
        "maximum_equilibrium_residual": maximum_residual,
        "maximum_contraction_bound": float(maximum_contraction),
        "maximum_newton_steps": maximum_newton_steps,
        "maximum_training_risk_increase": maximum_risk_increase,
        "wall_seconds": time.perf_counter() - clock_start,
        "validation": validation,
    }
    return {key: np.asarray(value) for key, value in recorded.items()}, statistics


def plot(all_trajectories, epsilons, destination):
    plt.style.use(ROOT / "paper.mplstyle")
    palette = ["#1764ab", "#d27713", "#238b45"]
    for field, stem, ylabel in [
        ("parameter_error", "parameters", r"$\|\theta(t)-\xi\|_2$"),
        ("eval_risk", "risk", "Held-out prediction risk"),
    ]:
        fig, ax = plt.subplots(figsize=(4.2, 3.15))
        for epsilon, color in zip(epsilons, palette):
            curves = all_trajectories[epsilon]
            iterations = curves[0]["updates"]
            values = np.stack([curve[field] for curve in curves])
            mean = np.mean(values, axis=0)
            denominator = int(round(1 / epsilon))
            ax.semilogy(
                iterations, mean, color=color, label=rf"$\varepsilon=1/{denominator}$"
            )

            ax.fill_between(
                iterations,
                np.min(values, axis=0),
                np.max(values, axis=0),
                color=color,
                alpha=0.17,
                linewidth=0,
            )
        ax.set_xlabel("Gradient-descent updates")
        ax.set_ylabel(ylabel)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(3, 3), useMathText=True)
        ax.legend(frameon=False)
        fig.tight_layout()
        for extension in ["pdf", "png"]:
            fig.savefig(destination / f"new-nonlinearity-{stem}.{extension}", dpi=400)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Weak-nonlinearity experiment")
    parser.add_argument("--updates", type=int, default=60000)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[1729, 1730, 1731, 1732, 1733]
    )
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--record-every", type=int, default=100)
    parser.add_argument("--output", type=Path, default=RESULTS)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Redraw existing results without training",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    epsilons = [0.5, 0.25, 0.125]
    if args.plot_only:
        stored_config = json.loads((args.output / "config.json").read_text())
        all_trajectories = {epsilon: [] for epsilon in epsilons}
        for seed in stored_config["seeds"]:
            for epsilon in epsilons:
                with np.load(args.output / f"seed-{seed}-eps-{epsilon:g}.npz") as data:
                    all_trajectories[epsilon].append(
                        {key: data[key] for key in data.files}
                    )
        plot(all_trajectories, epsilons, ROOT / "images")
        return
    config = {
        "description": "Exact implicit differentiation; fixed Gaussian empirical risk; independent held-out Gaussian risk",
        "dimension": 10,
        "n_train": 4096,
        "n_eval": 16384,
        "epsilons": epsilons,
        "seeds": args.seeds,
        "updates": args.updates,
        "step_size": args.step,
        "record_every": args.record_every,
        "solver": "Vectorized Newton on preactivation, original-equation residual checked at every update",
        "solver_tolerance": 1e-12,
        "arithmetic": "float64",
        "initialization": "xi + 0.1*(-xi1,1)/sqrt(2) + 0.03*(v,0), v random unit vector orthogonal to xi1",
        "pairing": "All epsilon values share training/evaluation inputs, xi and theta0 within each seed",
        "selection": "No tuning; horizon chosen from the initial target-Hessian timescales, not selected outcome curves",
        "uncertainty": "Plots show arithmetic mean with observed min-max seed range; summary reports sample standard deviation",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "blas_thread_environment": {
            name: os.environ.get(name)
            for name in [
                "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "OMP_NUM_THREADS",
            ]
        },
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    all_trajectories = {epsilon: [] for epsilon in epsilons}
    all_statistics = []
    start = time.perf_counter()
    for seed in args.seeds:
        x_train, x_eval, xi, theta0 = make_data(seed, 10, 4096, 16384)
        for epsilon in epsilons:
            trajectory, statistics = run_one(
                x_train,
                x_eval,
                xi,
                theta0,
                epsilon,
                args.updates,
                args.step,
                args.record_every,
            )
            statistics["seed"] = seed
            file_stem = f"seed-{seed}-eps-{epsilon:g}"
            np.savez_compressed(
                args.output / f"{file_stem}.npz", **trajectory, xi=xi, theta0=theta0
            )
            (args.output / f"{file_stem}.json").write_text(
                json.dumps(statistics, indent=2) + "\n"
            )
            all_trajectories[epsilon].append(trajectory)
            all_statistics.append(statistics)
            print(json.dumps(statistics), flush=True)
    summary = {
        "wall_seconds": time.perf_counter() - start,
        "failed_runs": 0,
        "runs": all_statistics,
        "aggregate": [],
    }
    for epsilon in epsilons:
        records = [record for record in all_statistics if record["epsilon"] == epsilon]
        aggregate = {"epsilon": epsilon}
        for key in [
            "final_parameter_error",
            "final_training_risk",
            "final_evaluation_risk",
            "wall_seconds",
        ]:
            values = np.array([record[key] for record in records])
            aggregate[key + "_mean"] = float(np.mean(values))
            aggregate[key + "_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else None
            )
        summary["aggregate"].append(aggregate)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if len(args.seeds) > 1:
        (ROOT / "images").mkdir(exist_ok=True)
        plot(all_trajectories, epsilons, ROOT / "images")
    print(
        json.dumps(
            {"aggregate": summary["aggregate"], "wall_seconds": summary["wall_seconds"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
