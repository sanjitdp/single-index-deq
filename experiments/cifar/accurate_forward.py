import torch


class ForwardSolveError(RuntimeError):
    def __init__(self, stats):
        self.stats = stats
        super().__init__(
            f"Forward residual requirement failed; no update allowed: {stats}"
        )


def residuals(func, state):
    value = func(state)
    error = (value - state).flatten(1).norm(dim=1)
    return error / (value.flatten(1).norm(dim=1) + 1e-9)


def solve_accurately(
    func,
    initial,
    solver,
    iterations,
    tolerance,
    limit,
    fallback=None,
    newton=None,
    policy="standard",
):

    state = initial
    calls = 0
    best = initial.clone()
    best_relative = initial.new_full((len(initial),), float("inf"))

    class EnoughAccuracy(Exception):
        pass

    class NonfiniteIterate(Exception):
        pass

    def counted(z):
        nonlocal calls, best, best_relative
        calls += 1
        if not torch.isfinite(z).all():
            raise NonfiniteIterate()
        value = func(z)
        relative = (value - z).flatten(1).norm(dim=1) / (
            value.flatten(1).norm(dim=1) + 1e-9
        )
        improved = torch.isfinite(relative) & (relative < best_relative)

        best = (
            torch.where(improved.reshape(-1, *([1] * (z.ndim - 1))), z, best)
            .detach()
            .clone()
        )
        best_relative = torch.where(improved, relative, best_relative).detach()
        if best_relative.max().item() <= tolerance:
            raise EnoughAccuracy()
        if not torch.isfinite(value).all():
            raise NonfiniteIterate()
        return value

    if policy not in ["standard", "strong"]:
        raise ValueError(policy)
    history = 200 if policy == "strong" else 40
    attempts = [
        (solver, {"LBFGS_thres": history, "ls": False}, iterations),
        (solver, {"LBFGS_thres": history, "ls": True}, iterations),
    ]
    if fallback is not None:

        attempts.append(
            (
                fallback,
                {
                    "m": 20 if policy == "strong" else 6,
                    "lam": 1e-6 if policy == "strong" else 1e-4,
                    "tau": 0.5,
                },
                5 * iterations,
            )
        )
    if newton is not None:
        attempts.append((newton, {}, 30))
    for attempt, (method, options, budget) in enumerate(attempts):
        info = {}
        try:
            state, _, info = method(
                counted,
                best,
                max_iter=budget,
                tol=tolerance,
                stop_mode="rel",
                **options,
            )
            counted(state)
        except (EnoughAccuracy, NonfiniteIterate):
            pass
        state = best
        relative = residuals(func, state)
        calls += 1
        stats = {
            "forward_relative_residual": relative.mean().item(),
            "forward_max_relative_residual": relative.max().item(),
            "forward_function_evaluations": calls,
            "forward_retries": attempt,
            "forward_residual_limit": limit,
        }
        if torch.isfinite(relative).all() and relative.max().item() <= limit:
            return state, info, stats
    raise ForwardSolveError(stats)
