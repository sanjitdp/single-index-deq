import torch


def gmres(operator, rhs, tolerance=1e-10, restart=40, max_iterations=400, notify=None):

    if restart < 1 or max_iterations < 1 or tolerance <= 0:
        raise ValueError("GMRES budgets and tolerance must be positive")
    x = torch.zeros_like(rhs)
    denominator = rhs.norm().clamp_min(1e-30)
    calls = 0
    history = []
    loose = None
    if rhs.norm().item() == 0:
        return x, {
            "status": "converged",
            "relative_residual": 0.0,
            "iterations": 0,
            "operator_calls": 0,
            "trace": [],
            "tightening_solution_change": 0.0,
        }
    for offset in range(0, max_iterations, restart):
        residual = rhs - operator(x)
        calls += 1
        beta = residual.norm()
        if beta / denominator <= tolerance:
            break
        basis = [residual / beta]
        count = min(restart, max_iterations - offset)
        h = torch.zeros(count + 1, count, dtype=torch.float64)
        target = torch.zeros(count + 1, 1, dtype=torch.float64)
        target[0, 0] = beta.item()
        candidate = x
        for column in range(count):
            vector = operator(basis[column])
            calls += 1
            for _ in range(2):
                for row in range(column + 1):
                    coefficient = (basis[row] * vector).sum()
                    h[row, column] += coefficient.item()
                    vector = vector - coefficient * basis[row]
            length = vector.norm()
            h[column + 1, column] = length.item()
            basis.append(vector / length.clamp_min(1e-30))

            if (column + 1) % 5 and column + 1 != count and length.item() > 1e-14:
                continue
            coefficients = torch.linalg.lstsq(
                h[: column + 2, : column + 1], target[: column + 2], driver="gelsd"
            ).solution[:, 0]
            candidate = x.clone()
            for row, coefficient in enumerate(coefficients):
                candidate.add_(basis[row], alpha=coefficient.item())
            relative = ((rhs - operator(candidate)).norm() / denominator).item()
            calls += 1
            history.append(
                {"iterations": offset + column + 1, "relative_residual": relative}
            )
            if relative <= 1e-7 and loose is None:
                loose = candidate.clone()
            if relative <= tolerance or not torch.isfinite(candidate).all():
                break
        x = candidate
        if notify is not None:
            notify(history[-1])
        if history[-1]["relative_residual"] <= tolerance or not torch.isfinite(x).all():
            break
    residual = operator(x) - rhs
    calls += 1
    relative = (residual.norm() / denominator).item()
    return x.detach(), {
        "status": "converged" if relative <= tolerance else "failed_adjoint_residual",
        "relative_residual": relative,
        "iterations": history[-1]["iterations"] if history else 0,
        "operator_calls": calls,
        "trace": history,
        "tightening_solution_change": (
            None
            if loose is None
            else ((x - loose).norm() / x.norm().clamp_min(1e-30)).item()
        ),
    }


def attach_reference(
    func, equilibrium, stats, restart=40, max_iterations=400, notify=None
):
    state = equilibrium.detach().requires_grad_()
    value = func(state)
    output = equilibrium.detach() + (value - value.detach())

    def backward(gradient):
        def operator(vector):
            return (
                vector - torch.autograd.grad(value, state, vector, retain_graph=True)[0]
            )

        adjoint, info = gmres(
            operator,
            gradient,
            restart=restart,
            max_iterations=max_iterations,
            notify=notify,
        )
        info["restart"] = restart
        info["max_iterations"] = max_iterations
        residual = operator(adjoint) - gradient
        denominators = gradient.flatten(1).norm(dim=1)
        floor = gradient.norm().item() / len(gradient) ** 0.5 * 1e-12
        info["max_example_relative_residual"] = (
            (
                residual.flatten(1).norm(dim=1)
                / denominators.clamp_min(max(floor, 1e-30))
            )
            .max()
            .item()
        )
        info["accepted"] = (
            info["relative_residual"] <= 1e-10
            and info["max_example_relative_residual"] <= 1e-7
        )
        stats.update(info)
        return adjoint

    output.register_hook(backward)
    return output


def compare(update, reference):
    norm = update.norm().item()
    reference_norm = reference.norm().item()
    dot = (update * reference).sum().item()
    return {
        "norm": norm,
        "reference_norm": reference_norm,
        "dot_reference": dot,
        "cosine": dot / (norm * reference_norm) if norm * reference_norm > 0 else None,
        "norm_ratio": norm / reference_norm if reference_norm > 0 else None,
        "relative_error": (
            ((update - reference).norm().item() / reference_norm)
            if reference_norm > 0
            else None
        ),
        "first_order_descent": dot > 0,
    }
