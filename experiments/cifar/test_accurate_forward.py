import torch
from model import broyden_solver
from accurate_forward import solve_accurately, ForwardSolveError
from newton_forward import newton_solver


def main():
    torch.set_num_threads(2)
    x = torch.zeros(128, 3, dtype=torch.float64)
    func = lambda z: 0.5 * z + 1
    state, _, stats = solve_accurately(func, x, broyden_solver, 50, 1e-8, 1e-7)
    assert torch.allclose(state, torch.full_like(x, 2))
    assert stats["forward_max_relative_residual"] <= 1e-7
    attempts = []

    def retry_solver(f, z, **kwargs):
        attempts.append(kwargs["ls"])
        return (torch.full_like(z, 2) if kwargs["ls"] else z), [], {}

    _, _, stats = solve_accurately(func, x, retry_solver, 50, 1e-8, 1e-7)
    assert attempts == [False, True] and stats["forward_retries"] == 1

    def one_bad_example(f, z, **kwargs):
        state = torch.full_like(z, 2)
        state[0] = 0
        return state, [], {}

    try:
        solve_accurately(func, x, one_bad_example, 50, 1e-3, 0.01)
    except ForwardSolveError as error:
        assert error.stats["forward_relative_residual"] < 0.01
        assert error.stats["forward_max_relative_residual"] > 0.99
    else:
        raise AssertionError("Batch means must not hide a failed example")

    def nan_after_good(f, z, **kwargs):
        f(torch.full_like(z, 2.0 + 1e-8))
        f(torch.full_like(z, float("nan")))
        raise AssertionError("A nonfinite iterate must abort the attempt")

    _, _, stats = solve_accurately(func, x, nan_after_good, 50, 1e-12, 1e-7)
    assert stats["forward_max_relative_residual"] < 1e-7

    matrix = torch.tensor([[1.2, 0.3], [0.2, -0.5]], dtype=torch.float64)
    forcing = torch.tensor([[1.0, 2.0], [0.0, 0.0]], dtype=torch.float64)
    linear = lambda z: z @ matrix.T + forcing
    with torch.no_grad():
        state, _, _ = newton_solver(linear, torch.zeros_like(forcing), tol=1e-9)
    exact = torch.linalg.solve(torch.eye(2, dtype=torch.float64) - matrix, forcing.T).T
    assert torch.allclose(state, exact, atol=1e-7, rtol=1e-7)
    print("Accurate-forward tests passed", flush=True)


if __name__ == "__main__":
    main()
