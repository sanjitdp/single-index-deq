import json
from pathlib import Path
import torch
from model import COMMIT
from backward import attach_backward


def main():
    torch.set_num_threads(1)
    torch.manual_seed(14)
    dtype = torch.float64
    matrix = torch.tensor(
        [[0.2, -0.3, 0.1], [0.1, 0.1, -0.1], [-0.2, 0.2, 0.1]], dtype=dtype
    )
    parameter = torch.randn(3, dtype=dtype, requires_grad=True)
    direct = torch.randn(3, dtype=dtype, requires_grad=True)
    state = torch.linalg.solve(torch.eye(3, dtype=dtype) - matrix, parameter.detach())[
        None
    ]
    errors = []
    for damping in [1.0, 0.5]:
        for terms in [1, 2, 3, 4, 5, 8]:
            parameter.grad = direct.grad = None
            output = attach_backward(
                lambda z: z @ matrix.T + parameter, state, "neumann", terms, damping
            )
            assert torch.equal(output.detach(), state)
            (output * direct).sum().backward()
            damped = (1 - damping) * torch.eye(3, dtype=dtype) + damping * matrix.T
            expected = torch.zeros(3, dtype=dtype)
            term = direct.detach()
            for _ in range(terms):
                expected += damping * term
                term = damped @ term
            error = (parameter.grad - expected).abs().max().item()
            assert error < 1e-12
            assert torch.equal(direct.grad, state[0])
            errors.append({"terms": terms, "damping": damping, "max_abs_error": error})
    parameter.grad = direct.grad = None
    stats = {}
    output = attach_backward(
        lambda z: z @ matrix.T + parameter,
        state,
        "ift",
        backward_tolerance=1e-12,
        stats=stats,
    )
    (output * direct).sum().backward()
    exact = torch.linalg.solve(torch.eye(3, dtype=dtype) - matrix.T, direct.detach())
    ift_error = (parameter.grad - exact).abs().max().item()
    assert ift_error < 1e-10

    injection = torch.tensor([[0.8, -0.2], [0.1, 0.5], [-0.3, 0.4]], dtype=dtype)
    nonlinear_parameter = torch.tensor([0.3, -0.4], dtype=dtype, requires_grad=True)
    frozen = torch.tensor([[0.1, -0.2, 0.5]], dtype=dtype)
    nonlinear_errors = []
    preactivation = frozen[0] @ matrix.T + injection @ nonlinear_parameter.detach()
    derivative = torch.diag(1 - torch.tanh(preactivation).square())
    jacobian = derivative @ matrix
    parameter_jacobian = derivative @ injection
    cotangent = torch.tensor([0.7, -0.8, 0.2], dtype=dtype)
    for damping in [1.0, 0.5]:
        for terms in [1, 2, 8]:
            nonlinear_parameter.grad = None
            output = attach_backward(
                lambda z: torch.tanh(z @ matrix.T + injection @ nonlinear_parameter),
                frozen,
                "neumann",
                terms,
                damping,
            )
            assert torch.equal(output.detach(), frozen)
            (output * cotangent).sum().backward()
            term = cotangent
            expected = torch.zeros(2, dtype=dtype)
            damped = (1 - damping) * torch.eye(3, dtype=dtype) + damping * jacobian.T
            for _ in range(terms):
                expected += damping * parameter_jacobian.T @ term
                term = damped @ term
            error = (nonlinear_parameter.grad - expected).abs().max().item()
            assert error < 1e-12
            nonlinear_errors.append(
                {"terms": terms, "damping": damping, "max_abs_error": error}
            )
    result = {
        "status": "passed",
        "neumann": errors,
        "ift_max_abs_error": ift_error,
        "ift_stats": stats,
        "nonlinear_nonequilibrium_neumann": nonlinear_errors,
        "upstream_commit": COMMIT,
    }
    print(json.dumps(result, indent=2))
    output_path = (
        Path(__file__).resolve().parents[1] / "results/cifar/backward_tests.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
