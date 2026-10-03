import torch


def attach_backward(
    func,
    equilibrium,
    method="neumann",
    terms=1,
    damping=1.0,
    backward_iterations=50,
    backward_tolerance=1e-6,
    stats=None,
):

    if terms < 1 or not 0 < damping <= 1:
        raise ValueError("Require terms >= 1 and 0 < damping <= 1")
    state = equilibrium.detach().requires_grad_()
    value = func(state)
    scale = damping if method == "neumann" else 1.0
    output = equilibrium.detach() + scale * (value - value.detach())

    def backward(gradient):
        def jt(vector):
            return torch.autograd.grad(value, state, vector, retain_graph=True)[0]

        if method == "neumann":
            term = gradient
            total = term
            for _ in range(terms - 1):
                term = (1.0 - damping) * term + damping * jt(term)
                total = total + term
            return total
        if method != "ift":
            raise ValueError(method)
        from torchdeq.solver import broyden_solver

        adjoint, _, info = broyden_solver(
            lambda v: jt(v) + gradient,
            torch.zeros_like(gradient),
            max_iter=backward_iterations,
            tol=backward_tolerance,
            stop_mode="rel",
        )
        if stats is not None:
            stats["backward_relative_residual"] = float(
                (
                    (adjoint - jt(adjoint) - gradient).norm()
                    / gradient.norm().clamp_min(1e-12)
                )
                .detach()
                .cpu()
            )
        return adjoint

    output.register_hook(backward)
    return output
