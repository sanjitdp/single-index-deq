from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "mpl-approximate"))
for variable in ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import brentq

METHODS = ("exact", "jfb", "neumann2", "neumann3", "damped2")
LABELS = ("Exact", "JFB (1 term)", "2 terms", "3 terms", r"2 terms, $\lambda=1/2$")
COLORS = ("#111111", "#0072B2", "#D55E00", "#009E73", "#CC79A7")
STYLES = ("-", "--", "-.", ":", (0, (5, 2, 1, 2)))
OUT = ROOT / "results" / "approximate"


def backward_factor(a, method):

    if method == "exact":
        return 1.0 / (1.0 - a)
    if method == "jfb":
        return np.ones_like(a)
    if method == "neumann2":
        return 1.0 + a
    if method == "neumann3":
        return 1.0 + a + a * a
    if method == "damped2":
        damping = 0.5
        return damping * (1.0 + (1.0 - damping) + damping * a)
    raise ValueError(method)


def equilibrium(u, b, initial=None, tolerance=1e-12, max_iterations=80):

    if abs(b) >= 1:
        raise RuntimeError(f"forward uniqueness guard violated: b={b}")
    lower = np.full_like(u, -1.0)
    upper = np.full_like(u, 1.0)
    y = np.tanh(u) if initial is None else np.clip(initial, -1.0, 1.0).copy()
    for iteration in range(max_iterations):
        value = np.tanh(u + b * y)
        residual = y - value
        maximum = float(np.max(np.abs(residual)))
        if maximum <= tolerance:
            return y, maximum, iteration + 1
        positive = residual > 0
        lower = np.where(positive, lower, y)
        upper = np.where(positive, y, upper)
        proposal = y - residual / (1.0 - b * (1.0 - value * value))
        invalid = (proposal < lower) | (proposal > upper) | ~np.isfinite(proposal)
        y = np.where(invalid, (lower + upper) / 2, proposal)
    raise RuntimeError(f"forward solver failed: residual={maximum}")


def nonlinear_gradient(x, target, theta, method, warm=None):
    u = x @ theta[:-1]
    y, residual, iterations = equilibrium(u, theta[-1], warm)

    sigma = np.tanh(u + theta[-1] * y)
    derivative = 1.0 - sigma * sigma
    weight = (
        2.0
        * (y - target)
        * derivative
        * backward_factor(theta[-1] * derivative, method)
    )
    gradient = np.r_[x.T @ weight, y @ weight] / len(x)
    return gradient, y, float(np.mean((y - target) ** 2)), residual, iterations


def checks():
    rng = np.random.default_rng(90210)
    x = rng.normal(size=(73, 5))
    xi = rng.normal(size=5)
    xi /= np.linalg.norm(xi)
    theta = np.r_[xi + rng.normal(size=5) * 0.03, 0.15]
    target = np.tanh(x @ xi)
    gradient, y, risk, residual, _ = nonlinear_gradient(x, target, theta, "exact")
    differences = []
    step = 1e-6
    for j in range(len(theta)):
        plus, minus = theta.copy(), theta.copy()
        plus[j] += step
        minus[j] -= step
        fp = nonlinear_gradient(x, target, plus, "exact")[2]
        fm = nonlinear_gradient(x, target, minus, "exact")[2]
        differences.append((fp - fm) / (2 * step))
    finite_difference_error = float(np.max(np.abs(gradient - differences)))
    roots = np.array(
        [
            brentq(lambda z: z - np.tanh(u + theta[-1] * z), -1, 1, xtol=1e-14)
            for u in x @ theta[:-1]
        ]
    )
    brent_error = float(np.max(np.abs(y - roots)))
    assert finite_difference_error < 1e-8
    assert brent_error < 2e-12

    a = np.linspace(-0.9, 0.9, 17)
    assert np.allclose(
        backward_factor(a, "damped2"), sum(0.5 * (0.5 + 0.5 * a) ** j for j in range(2))
    )
    exact = linear_gradient(np.array([np.sqrt(8), -0.3]), "exact")
    for m, method in enumerate(("jfb", "neumann2", "neumann3"), 1):
        assert np.allclose(
            linear_gradient(np.array([np.sqrt(8), -0.3]), method),
            (1 - (-0.3) ** m) * exact,
        )
    stress_u = np.r_[np.linspace(-10, 10, 101), np.linspace(-0.001, 0.001, 101)]
    stress_error = 0.0
    for b in (-0.99, -0.5, 0.0, 0.5, 0.99):
        candidate, _, _ = equilibrium(stress_u, b, initial=np.sin(100 * stress_u))
        independent = np.array(
            [
                brentq(lambda z: z - np.tanh(u + b * z), -1, 1, xtol=1e-14)
                for u in stress_u
            ]
        )
        stress_error = max(stress_error, float(np.max(np.abs(candidate - independent))))
    assert stress_error < 1e-10
    return {
        "finite_difference_max_absolute_error": finite_difference_error,
        "independent_brent_max_difference": brent_error,
        "forward_residual": residual,
        "damped_polynomial_identity": True,
        "linear_polynomial_identity": True,
        "stress_root_cases": 1010,
        "stress_root_max_difference": stress_error,
    }


def linear_gradient(theta, method):
    c, b = theta
    coefficient = c / (1 - b)
    return (
        2
        * (coefficient - 1)
        * backward_factor(b, method)
        * np.array([1.0, coefficient])
    )


def run_linear(steps=5000, eta=0.02):

    initializations = (
        np.array([np.sqrt(8), 0.0]),
        np.array([np.sqrt(8) - 0.025, 0.0]),
        np.array([np.sqrt(8) + 0.025, 0.0]),
        np.array([np.sqrt(8), -0.025]),
        np.array([np.sqrt(8), 0.025]),
    )
    risk = np.empty((len(initializations), len(METHODS), steps + 1))
    theta_history = np.empty((*risk.shape, 2))
    timings = np.empty((len(initializations), len(METHODS)))
    for i, theta0 in enumerate(initializations):
        for j, method in enumerate(METHODS):
            start = time.perf_counter()
            theta = theta0.copy()
            for t in range(steps + 1):
                if theta[1] >= 1 or not np.all(np.isfinite(theta)):
                    raise RuntimeError(
                        f"linear forward branch failure: {i}, {method}, {t}"
                    )
                risk[i, j, t] = (theta[0] / (1 - theta[1]) - 1) ** 2
                theta_history[i, j, t] = theta
                if t < steps:
                    theta -= eta * linear_gradient(theta, method)
            timings[i, j] = time.perf_counter() - start
    np.savez_compressed(
        OUT / "linear.npz",
        risk=risk,
        theta=theta_history,
        runtime_seconds=timings,
        initializations=initializations,
        eta=eta,
        steps=steps,
        methods=METHODS,
    )
    summary = {
        "steps": steps,
        "eta": eta,
        "initializations": [x.tolist() for x in initializations],
        "methods": {
            m: {
                "final_risk": float(risk[0, j, -1]),
                "final_theta": theta_history[0, j, -1].tolist(),
                "nearby_final_risks": risk[1:, j, -1].tolist(),
                "runtime_seconds": float(timings[0, j]),
            }
            for j, m in enumerate(METHODS)
        },
    }
    (OUT / "linear_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def run_nonlinear(seeds, steps=40000, eta=0.2, n=8000, evaluate_every=20):
    summaries = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        xi = rng.normal(size=20)
        xi /= np.linalg.norm(xi)
        delta = rng.normal(size=20)
        delta *= 0.2 / np.linalg.norm(delta)
        theta0 = np.r_[xi + delta, 0.15]
        x = rng.normal(size=(n, 20))
        x_eval = rng.normal(size=(n, 20))
        target, target_eval = np.tanh(x @ xi), np.tanh(x_eval @ xi)
        eval_steps = np.unique(np.r_[np.arange(0, steps + 1, evaluate_every), steps])
        for method in METHODS:
            start = time.perf_counter()
            theta, warm = theta0.copy(), None
            train_risk = np.full(steps + 1, np.nan)
            residuals = np.full(steps + 1, np.nan)
            solver_iterations = np.zeros(steps + 1, dtype=np.int16)
            theta_history = np.full((len(eval_steps), 21), np.nan)
            eval_risk = np.full(len(eval_steps), np.nan)
            parameter_error = np.full(len(eval_steps), np.nan)
            max_eval_residual = 0.0
            max_abs_recurrence = 0.0
            status = "completed"
            error = None
            eval_index = 0
            try:
                for t in range(steps + 1):
                    gradient, y, risk, residual, n_iterations = nonlinear_gradient(
                        x, target, theta, method, warm
                    )
                    warm = y
                    train_risk[t] = risk
                    residuals[t] = residual
                    solver_iterations[t] = n_iterations
                    max_abs_recurrence = max(max_abs_recurrence, abs(float(theta[-1])))
                    if t == eval_steps[eval_index]:
                        pred, res, _ = equilibrium(x_eval @ theta[:-1], theta[-1])
                        eval_risk[eval_index] = np.mean((pred - target_eval) ** 2)
                        theta_history[eval_index] = theta
                        parameter_error[eval_index] = np.linalg.norm(
                            theta - np.r_[xi, 0.0]
                        )
                        max_eval_residual = max(max_eval_residual, res)
                        if eval_index < len(eval_steps) - 1:
                            eval_index += 1
                    if t < steps:
                        theta -= eta * gradient
            except Exception as exc:
                status, error = "failed", repr(exc)
            runtime = time.perf_counter() - start
            np.savez_compressed(
                OUT / f"nonlinear_seed{seed}_{method}.npz",
                train_risk=train_risk,
                eval_risk=eval_risk,
                parameter_error=parameter_error,
                theta=theta_history,
                eval_steps=eval_steps,
                residual=residuals,
                solver_iterations=solver_iterations,
                xi=xi,
                theta0=theta0,
                seed=seed,
                eta=eta,
                steps=steps,
                n_train=n,
                n_eval=n,
                runtime_seconds=runtime,
            )
            row = {
                "seed": seed,
                "method": method,
                "status": status,
                "error": error,
                "initial_risk": float(train_risk[0]),
                "final_train_risk": float(train_risk[-1]),
                "final_eval_risk": float(eval_risk[-1]),
                "final_parameter_error": float(parameter_error[-1]),
                "max_train_residual": float(np.nanmax(residuals)),
                "max_eval_residual": max_eval_residual,
                "max_abs_theta2": max_abs_recurrence,
                "runtime_seconds": runtime,
            }
            summaries.append(row)
            (OUT / "nonlinear_runs.json").write_text(
                json.dumps(summaries, indent=2) + "\n"
            )
            print(json.dumps(row), flush=True)
    aggregate = {}
    for method in METHODS:
        rows = [row for row in summaries if row["method"] == method]
        aggregate[method] = {
            "completed": sum(row["status"] == "completed" for row in rows),
            "attempted": len(rows),
        }
        for key in (
            "final_train_risk",
            "final_eval_risk",
            "final_parameter_error",
            "runtime_seconds",
        ):
            values = np.array([row[key] for row in rows])
            aggregate[method][key] = {
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
    (OUT / "nonlinear_summary.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    if any(row["status"] != "completed" for row in summaries):
        raise RuntimeError(
            "Some nonlinear runs failed; retained individual results and summaries."
        )
    return aggregate


def plot(seeds):
    style = ROOT / "paper.mplstyle"
    if style.exists():
        plt.style.use(style)
    else:
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.size": 12,
                "axes.labelsize": 13,
                "xtick.labelsize": 11,
                "ytick.labelsize": 11,
                "legend.fontsize": 10,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        )
    linear = np.load(OUT / "linear.npz")
    for panel in ("linear", "nonlinear"):
        fig, ax = plt.subplots(figsize=(4.2, 3.15))
        for j, (method, label, color, style) in enumerate(
            zip(METHODS, LABELS, COLORS, STYLES)
        ):
            if panel == "linear":
                values = linear["risk"][0, j]

                displayed = np.where(values >= 1e-14, values, np.nan)
                ax.semilogy(
                    np.arange(len(values)),
                    displayed,
                    label=label,
                    color=color,
                    linestyle=style,
                    linewidth=1.6,
                )
            else:
                traces = [
                    np.load(OUT / f"nonlinear_seed{seed}_{method}.npz")["train_risk"]
                    for seed in seeds
                ]
                values = np.array(traces)
                mean = values.mean(axis=0)
                sd = (
                    values.std(axis=0, ddof=1)
                    if len(seeds) > 1
                    else np.zeros(values.shape[1])
                )
                iterations = np.arange(len(mean))
                ax.semilogy(
                    iterations,
                    np.where(mean >= 1e-14, mean, np.nan),
                    label=label,
                    color=color,
                    linestyle=style,
                    linewidth=1.6,
                )
                ax.fill_between(
                    iterations,
                    np.maximum(mean - sd, 1e-14),
                    np.maximum(mean + sd, 1e-14),
                    where=mean >= 1e-14,
                    color=color,
                    alpha=0.10,
                    linewidth=0,
                )
            ax.set_xlabel("Gradient-descent updates")
            ax.set_ylabel("Training risk")
            ax.grid(True, which="major")
            ax.set_ylim(bottom=1e-14)
        ax.legend(
            loc="lower right" if panel == "linear" else "upper right", frameon=False
        )
        fig.tight_layout()
        (ROOT / "images").mkdir(exist_ok=True)
        for extension in ("pdf", "png"):
            fig.savefig(
                ROOT / "images" / f"new-approximate-{panel}.{extension}",
                dpi=400,
                bbox_inches="tight",
            )
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104]
    )
    parser.add_argument("--n", type=int, default=8000)
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Save separate pilot data, never overwrite final runs",
    )
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run analytic and independent numerical checks without training",
    )
    args = parser.parse_args()
    if args.check_only:
        print(json.dumps(checks(), indent=2))
        return
    global OUT
    if args.pilot:
        OUT = OUT / "pilot"
    OUT.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        plot(args.seeds)
        return
    start = time.perf_counter()
    config = {
        "seeds": args.seeds,
        "steps": args.steps,
        "eta": args.eta,
        "n_train": args.n,
        "n_eval": args.n,
        "d": 20,
        "initial_weight_error": 0.2,
        "initial_theta2": 0.15,
        "evaluation_stride": 20,
        "forward_tolerance": 1e-12,
        "methods": METHODS,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "environment_threads": {
            key: os.environ.get(key)
            for key in (
                "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "OMP_NUM_THREADS",
            )
        },
    }
    config["pilot_design_decision"] = (
        "Seed 100, 1000 updates, common eta=.2, all five methods; "
        "no failures, exact final parameter error .04768 and risk 1.883e-6. "
        "Target Hessian smallest eigenvalue .001579. Fixed final horizon "
        "40000 chosen before running remaining seeds to resolve the slow direction; "
        "no method-specific step tuning or held-out-data selection."
    )
    (OUT / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    validation = checks()
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print("Validation:", validation, flush=True)
    run_linear()
    run_nonlinear(args.seeds, args.steps, args.eta, args.n)
    if not args.pilot:
        plot(args.seeds)
    print(f"Total wall time: {time.perf_counter()-start:.3f} seconds", flush=True)


if __name__ == "__main__":
    main()
