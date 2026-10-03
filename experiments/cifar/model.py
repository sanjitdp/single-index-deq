import argparse
from pathlib import Path
import sys
import platform
import hashlib

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / "third_party" / "torchdeq"
COMMIT = "4f6bd5fa66dd991cad74fcc847c88061764cf8db"
sys.path[:0] = [str(UPSTREAM), str(UPSTREAM / "deq-zoo/mdeq/mdeq_cifar/mdeq")]

import torch
import yaml
from torchdeq.utils import add_deq_args
from torchdeq.utils.layer_utils import MDEQWrapper
from torchdeq.solver import broyden_solver, anderson_solver
from torchdeq.grad import backward_factory
from models.mdeq import get_cls_net
from backward import attach_backward
from accurate_forward import solve_accurately, residuals, ForwardSolveError
from newton_forward import newton_solver


class ControlledDEQ(torch.nn.Module):
    def __init__(
        self,
        method,
        terms,
        damping,
        forward_iterations=22,
        forward_tolerance=1e-3,
        backward_iterations=50,
        backward_tolerance=1e-6,
        forward_residual_limit=None,
        forward_policy="standard",
    ):
        super().__init__()
        self.method, self.terms, self.damping = method, terms, damping
        self.forward_iterations = forward_iterations
        self.forward_tolerance = forward_tolerance
        self.backward_iterations = backward_iterations
        self.backward_tolerance = backward_tolerance
        self.forward_residual_limit = forward_residual_limit
        self.forward_policy = forward_policy
        self.stats = {}

    def forward(self, func, initial, **kwargs):
        wrapped = MDEQWrapper(func, initial)
        with torch.no_grad():
            if self.forward_residual_limit is not None:
                state, info, self.stats = solve_accurately(
                    wrapped,
                    wrapped.list2vec(initial),
                    broyden_solver,
                    self.forward_iterations,
                    self.forward_tolerance,
                    self.forward_residual_limit,
                    fallback=anderson_solver,
                    newton=newton_solver,
                    policy=self.forward_policy,
                )
            else:
                state, _, info = broyden_solver(
                    wrapped,
                    wrapped.list2vec(initial),
                    max_iter=self.forward_iterations,
                    tol=self.forward_tolerance,
                    stop_mode="abs",
                )
                selected_relative = info["rel_trace"].gather(
                    1, info["nstep"].long()[:, None]
                )
                self.stats = {
                    "forward_relative_residual": float(selected_relative.mean().cpu()),
                    "forward_lowest_relative_residual": float(
                        info["rel_lowest"].mean().cpu()
                    ),
                    "forward_selected_step": float(info["nstep"].float().mean().cpu()),
                }
        if self.training:
            if self.method == "phantom":

                state = backward_factory(self.terms, tau=self.damping)(
                    self, wrapped, state.detach()
                )[0]
                if self.forward_residual_limit is not None:
                    with torch.no_grad():
                        relative = residuals(wrapped, state.detach())
                    self.stats["phantom_output_max_relative_residual"] = (
                        relative.max().item()
                    )
                    if (
                        not torch.isfinite(relative).all()
                        or relative.max().item() > self.forward_residual_limit
                    ):
                        raise ForwardSolveError(self.stats)
            else:
                state = attach_backward(
                    wrapped,
                    state,
                    self.method,
                    self.terms,
                    self.damping,
                    self.backward_iterations,
                    self.backward_tolerance,
                    stats=self.stats,
                )
        return [wrapped.vec2list(state)], info


def make_model(
    classes=10,
    method="neumann",
    terms=1,
    damping=1.0,
    forward_iterations=22,
    forward_tolerance=1e-3,
    backward_iterations=50,
    backward_tolerance=1e-6,
    packaged_forward=False,
    forward_residual_limit=None,
    initial_state_gain=1.0,
    initial_branch_gain=None,
    forward_policy="standard",
):
    cfg_file = UPSTREAM / "deq-zoo/mdeq/mdeq_cifar/configs/small.yaml"
    with cfg_file.open() as stream:
        cfg = yaml.safe_load(stream)
    cfg["MODEL"]["NUM_CLASSES"] = classes
    parser = argparse.ArgumentParser()
    add_deq_args(parser)
    args = parser.parse_args(
        [
            "--norm_type",
            "weight_norm",
            "--f_solver",
            "broyden",
            "--f_max_iter",
            "22",
            "--grad",
            "5",
            "--tau",
            "0.5",
        ]
    )
    model = get_cls_net(cfg, args)
    if not 0 < initial_state_gain <= 1:
        raise ValueError("Initial state gain must be in (0,1]")

    with torch.no_grad():
        for layer in model.full_stage.post_fuse_layers:
            layer.gnorm.weight.mul_(initial_state_gain)
        if initial_branch_gain is not None:
            if not 0 < initial_branch_gain <= 1:
                raise ValueError("Initial branch gain must be in (0,1]")
            for branch in model.full_stage.branches:
                for block in branch.blocks:
                    block.gn3.weight.mul_(initial_branch_gain)
                    block.gn3.bias.fill_(1.0)
    if not packaged_forward:
        model.deq = ControlledDEQ(
            method,
            terms,
            damping,
            forward_iterations,
            forward_tolerance,
            backward_iterations,
            backward_tolerance,
            forward_residual_limit,
            forward_policy,
        )
    return model


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
