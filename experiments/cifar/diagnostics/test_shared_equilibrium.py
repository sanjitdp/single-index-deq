from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from backward import attach_backward
from shared_equilibrium_math import gmres, attach_reference, compare


def main():
    torch.set_num_threads(1)
    torch.manual_seed(41)
    dtype = torch.float64

    jacobian = torch.tensor(
        [[-1.0, 2.0, 0.3], [0.0, -1.2, 0.2], [0.0, 0.0, 0.4]], dtype=dtype
    )
    matrix = torch.eye(3, dtype=dtype) - jacobian
    rhs = torch.randn(4, 3, dtype=dtype)
    solution, report = gmres(lambda v: v @ matrix, rhs, restart=8)
    expected = torch.linalg.solve(matrix.T, rhs.T).T
    assert report["status"] == "converged"
    torch.testing.assert_close(solution, expected, atol=1e-11, rtol=1e-11)
    diagonal = torch.linspace(1.0, 2.0, 12, dtype=dtype)
    long_rhs = torch.randn(2, 12, dtype=dtype)
    restarted, report = gmres(
        lambda v: diagonal * v, long_rhs, restart=5, max_iterations=100
    )
    assert report["status"] == "converged" and report["iterations"] > 5
    torch.testing.assert_close(restarted, long_rhs / diagonal, atol=1e-9, rtol=1e-9)

    hard_diagonal = torch.logspace(0.0, 2.0, 32, dtype=dtype)
    hard_rhs = torch.randn(2, 32, dtype=dtype)
    _, capped = gmres(
        lambda v: hard_diagonal * v, hard_rhs, restart=5, max_iterations=5
    )
    assert capped["status"] == "failed_adjoint_residual"
    progress = []
    improved, improved_info = gmres(
        lambda v: hard_diagonal * v,
        hard_rhs,
        restart=40,
        max_iterations=2000,
        notify=progress.append,
    )
    assert improved_info["status"] == "converged" and progress
    torch.testing.assert_close(improved, hard_rhs / hard_diagonal, atol=1e-9, rtol=1e-9)
    _, failure = gmres(lambda v: torch.zeros_like(v), rhs, max_iterations=5)
    assert failure["status"] == "failed_adjoint_residual"
    zero, report = gmres(lambda v: v @ matrix, torch.zeros_like(rhs))
    assert report["status"] == "converged" and not zero.any()
    parameter = torch.randn(3, dtype=dtype, requires_grad=True)
    head = torch.randn(3, dtype=dtype, requires_grad=True)
    equilibrium = torch.linalg.solve(matrix, parameter.detach())[None]
    stats = {}
    output = attach_reference(
        lambda z: z @ jacobian.T + parameter,
        equilibrium,
        stats,
        restart=100,
        max_iterations=2000,
    )
    assert torch.equal(output.detach(), equilibrium)
    (output * head).sum().backward()
    torch.testing.assert_close(
        parameter.grad, torch.linalg.solve(matrix.T, head.detach())
    )
    torch.testing.assert_close(head.grad, equilibrium[0])
    assert stats["accepted"]
    assert stats["restart"] == 100 and stats["max_iterations"] == 2000

    for terms, damping, expected_scale in [
        (1, 1.0, 1.0),
        (2, 1.0, 0.0),
        (3, 1.0, 1.0),
        (2, 0.5, 0.5),
        (5, 0.5, 0.5),
    ]:
        parameter.grad = None
        state = parameter.detach()[None] / 2
        result = attach_backward(
            lambda z: -z + parameter, state, "neumann", terms, damping
        )
        assert torch.equal(result.detach(), state)
        result.sum().backward()
        torch.testing.assert_close(
            parameter.grad, torch.full_like(parameter, expected_scale)
        )
    assert compare(torch.zeros(3), torch.ones(3))["cosine"] is None

    from shared_equilibrium import Fixed, CASES

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.injection = torch.nn.Linear(3, 3, dtype=dtype)
            self.head = torch.nn.BatchNorm1d(3, dtype=dtype)
            self.deq = torch.nn.Identity()

        def forward(self, images):
            injection = self.injection(images)
            states, _ = self.deq(
                lambda z: [z @ jacobian.T + injection], [torch.zeros_like(images)]
            )
            return self.head(states[0][0])

    net = Toy().train()
    images = torch.randn(5, 3, dtype=dtype)
    labels = torch.tensor([0, 1, 2, 1, 0])
    original = {k: v.clone() for k, v in net.state_dict().items()}
    exact_state = torch.linalg.solve(matrix, net.injection(images).T).T
    exact_logits = net.head(exact_state)
    torch.nn.functional.cross_entropy(exact_logits, labels).backward()
    exact_gradients = torch.cat([p.grad.flatten() for p in net.parameters()])
    for case in CASES:
        net.load_state_dict(original)
        net.zero_grad(set_to_none=True)
        net.deq = Fixed(exact_state.detach(), case)
        logits = net(images)
        assert torch.equal(logits.detach(), exact_logits.detach())
        torch.nn.functional.cross_entropy(logits, labels).backward()
        if case[0] == "ift_reference":
            torch.testing.assert_close(
                torch.cat([p.grad.flatten() for p in net.parameters()]),
                exact_gradients,
                atol=1e-9,
                rtol=1e-9,
            )
            assert net.deq.stats["accepted"]
    print(
        "Shared-equilibrium tests passed: GMRES, audited IFT, unchanged values, cancellation, damping."
    )


if __name__ == "__main__":
    main()
