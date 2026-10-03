import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np
import torch
from model import make_model, state_digest, broyden_solver, anderson_solver
from torchvision import datasets, transforms
from backward import attach_backward
from capture import Capture, Captured
from shared_equilibrium_math import attach_reference, compare
from torchdeq.utils.layer_utils import MDEQWrapper

CASES = [
    ("ift_reference", 0, 1.0),
    ("jfb", 1, 1.0),
    ("two_terms", 2, 1.0),
    ("three_terms", 3, 1.0),
    ("damped_two", 2, 0.5),
    ("damped_five", 5, 0.5),
]


class Fixed(torch.nn.Module):
    def __init__(self, state, case=None, reference_options=None):
        super().__init__()
        self.state, self.case, self.stats = state, case, {}
        self.reference_options = reference_options or {}

    def forward(self, func, initial, **kwargs):
        wrapped = MDEQWrapper(func, initial)
        if self.case is None:
            value = self.state.detach()
        elif self.case[0] == "ift_reference":
            value = attach_reference(
                wrapped, self.state, self.stats, **self.reference_options
            )
        else:
            value = attach_backward(
                wrapped, self.state, "neumann", self.case[1], self.case[2]
            )
        return [wrapped.vec2list(value)], {}


class Indexed(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label, index


def solve(func, initial, notify):

    best = initial.detach().clone()
    best_relative = best.new_full((len(best),), float("inf"))
    states, stages = {}, []
    calls = 0
    started = time.perf_counter()

    class Stop(Exception):
        pass

    class Nonfinite(Exception):
        pass

    def evaluate(z):
        nonlocal best, best_relative, calls
        calls += 1
        if not torch.isfinite(z).all():
            raise Nonfinite()
        value = func(z)
        relative = (value - z).norm(dim=1) / (value.norm(dim=1) + 1e-9)
        improved = torch.isfinite(relative) & (relative < best_relative)
        best = torch.where(improved[:, None], z, best).detach().clone()
        best_relative = torch.where(improved, relative, best_relative)
        for tolerance in (1e-5, 1e-7):
            if tolerance not in states and best_relative.max().item() <= tolerance:
                states[tolerance] = best.clone()
        if best_relative.max().item() <= 1e-7:
            raise Stop()
        if not torch.isfinite(value).all():
            raise Nonfinite()
        return value

    ladder = [
        ("broyden", broyden_solver, dict(max_iter=1000, LBFGS_thres=200, ls=False)),
        (
            "broyden_linesearch",
            broyden_solver,
            dict(max_iter=1000, LBFGS_thres=200, ls=True),
        ),
        ("anderson", anderson_solver, dict(max_iter=5000, m=20, lam=1e-6, tau=0.5)),
    ]
    with torch.no_grad():
        for name, solver, options in ladder:
            try:
                result, _, _ = solver(
                    evaluate, best, tol=1e-7, stop_mode="rel", **options
                )
                evaluate(result)
            except (Stop, Nonfinite):
                pass
            value = func(best)
            residual = (value - best).norm(dim=1) / (value.norm(dim=1) + 1e-9)
            stages.append(
                {
                    "stage": name,
                    "max_relative_residual": residual.max().item(),
                    "calls": calls,
                    "seconds": time.perf_counter() - started,
                }
            )
            notify(stages[-1])
            if torch.isfinite(residual).all() and residual.max().item() <= 1e-7:
                break
        audited = {}
        for tolerance, state in states.items():
            value = func(state)
            residual = (value - state).norm(dim=1) / (value.norm(dim=1) + 1e-9)
            if torch.isfinite(residual).all() and residual.max().item() <= tolerance:
                audited[tolerance] = state
        return audited, {
            "stages": stages,
            "calls": calls,
            "accepted_tolerances": sorted(audited, reverse=True),
            "seconds": time.perf_counter() - started,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, choices=range(6), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mirror", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--adjoint-restart", type=int, default=40)
    parser.add_argument("--adjoint-iterations", type=int, default=400)
    parser.add_argument("--previous-report", type=Path)
    args = parser.parse_args()
    if args.adjoint_restart < 1 or args.adjoint_iterations < 1:
        parser.error("Adjoint budgets must be positive")
    if args.output.exists():
        raise RuntimeError(
            "Existing diagnostic is preserved; choose a new output for a retry"
        )
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(0)
    checkpoint_name = ["jfb", "two_terms"][args.index // 3]
    batch_number = args.index % 3
    checkpoint_path = args.checkpoints / checkpoint_name / "final.pt"
    started = time.perf_counter()
    report = {
        "status": "running",
        "checkpoint": str(checkpoint_path),
        "task": args.index,
        "batch_number": batch_number,
        "dtype": "float64",
        "gpu": torch.cuda.get_device_name(),
        "cases": CASES,
        "gradient_records": [],
        "steps": [],
        "events": [],
        "adjoint_solver": {
            "restart": args.adjoint_restart,
            "max_iterations": args.adjoint_iterations,
            "global_residual_limit": 1e-10,
            "max_example_residual_limit": 1e-7,
        },
        "previous_report": str(args.previous_report) if args.previous_report else None,
        "note": "Local diagnostics only; no optimizer trajectory or checkpoint is modified.",
    }

    def save():
        report["elapsed_seconds"] = time.perf_counter() - started
        for path in [args.output, args.mirror]:
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps(report, indent=2, allow_nan=False) + "\n"
                )
                temporary.replace(path)

    def notify(event):
        report["events"].append(event)
        print(json.dumps(event), flush=True)
        save()

    save()
    try:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        dataset = Indexed(
            datasets.CIFAR10(str(args.data), train=True, transform=transform)
        )
        permutation = np.random.default_rng(314159).permutation(len(dataset))
        loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, permutation[:45000]),
            batch_size=128,
            shuffle=True,
            generator=torch.Generator().manual_seed(10000),
            num_workers=0,
        )
        for number, (images, labels, indices) in enumerate(loader):
            if number == batch_number:
                break
        report["dataset_indices"] = indices.tolist()
        report["images_sha256"] = hashlib.sha256(images.numpy().tobytes()).hexdigest()
        images, labels = images.cuda().double(), labels.cuda()
        model = make_model()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        del checkpoint
        report["checkpoint_state_sha256_before_dtype_cast"] = state_digest(model)
        if args.previous_report:
            previous = json.loads(args.previous_report.read_text())
            fields = (
                "task",
                "batch_number",
                "dataset_indices",
                "images_sha256",
                "checkpoint_state_sha256_before_dtype_cast",
                "dtype",
            )
            report["paired_with_previous_report"] = all(
                report[k] == previous[k] for k in fields
            )
            if not report["paired_with_previous_report"]:
                raise RuntimeError(
                    "Retry differs from original checkpoint, images, task, or precision"
                )
        model = model.cuda().double().train()

        original = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        parameters = list(model.named_parameters())
        block_mask = torch.cat(
            [
                torch.full(
                    (parameter.numel(),),
                    name.startswith("full_stage."),
                    device="cuda",
                    dtype=torch.bool,
                )
                for name, parameter in parameters
            ]
        )
        parameter_norm = (
            torch.cat([p.detach().flatten() for _, p in parameters]).norm().item()
        )

        def restore():
            model.load_state_dict(original)
            model.zero_grad(set_to_none=True)

        def capture():
            module = Capture()
            model.deq = module
            with torch.no_grad():
                try:
                    model(images)
                except Captured:
                    pass
            return module.func, module.initial

        def loss_at(state):
            model.deq = Fixed(state)
            with torch.no_grad():
                logits = model(images)[-1]
                return torch.nn.functional.cross_entropy(logits, labels).item()

        restore()
        func, initial = capture()
        states, info = solve(
            func, initial, lambda row: notify({"baseline_forward": row})
        )
        report["baseline_forward"] = info
        del func, initial
        gradients, losses = {}, {}
        for tolerance, state in sorted(states.items(), reverse=True):
            baseline_logits = None
            rows = []
            reference_ok = False
            for case in CASES:
                restore()
                module = Fixed(
                    state,
                    case,
                    reference_options={
                        "restart": args.adjoint_restart,
                        "max_iterations": args.adjoint_iterations,
                        "notify": lambda row: notify(
                            {"adjoint_progress": row, "forward_tolerance": tolerance}
                        ),
                    },
                )
                model.deq = module
                logits = model(images)[-1]
                loss = torch.nn.functional.cross_entropy(logits, labels)
                loss.backward()
                gradient = torch.cat(
                    [
                        (
                            p.grad.detach().flatten()
                            if p.grad is not None
                            else torch.zeros_like(p).flatten()
                        )
                        for _, p in parameters
                    ]
                )
                row = {
                    "method": case[0],
                    "forward_tolerance": tolerance,
                    "loss": loss.item(),
                    "status": "measured",
                    "adjoint": module.stats,
                }
                if not torch.isfinite(gradient).all() or not torch.isfinite(loss):
                    row["status"] = "nonfinite_gradient"
                else:
                    gradients[(tolerance, case[0])] = gradient.detach().clone()
                if baseline_logits is None:
                    baseline_logits = logits.detach().clone()
                    reference_ok = module.stats.get("accepted", False)
                    losses[tolerance] = loss.item()
                row["max_logit_difference_between_methods"] = (
                    (logits.detach() - baseline_logits).abs().max().item()
                )
                if row["max_logit_difference_between_methods"] != 0:
                    raise RuntimeError(
                        "Backward methods did not receive identical forward predictions"
                    )
                rows.append(row)
                del loss, logits, gradient
                notify(
                    {
                        "gradient_done": case[0],
                        "tolerance": tolerance,
                        "reference_accepted": reference_ok,
                    }
                )
            for row in rows:
                key = (tolerance, row["method"])
                if reference_ok and key in gradients:
                    reference = gradients[(tolerance, "ift_reference")]
                    row["full_parameters"] = compare(gradients[key], reference)
                    row["equilibrium_parameters"] = compare(
                        gradients[key][block_mask], reference[block_mask]
                    )
                elif not reference_ok:
                    row["comparison_status"] = "no_accepted_implicit_reference"
            report["gradient_records"].extend(rows)
            save()
        report["forward_tolerance_sensitivity"] = []
        for name, _, _ in CASES:
            if (1e-5, name) in gradients and (1e-7, name) in gradients:
                report["forward_tolerance_sensitivity"].append(
                    {
                        "method": name,
                        "full_parameters": compare(
                            gradients[(1e-5, name)], gradients[(1e-7, name)]
                        ),
                        "equilibrium_parameters": compare(
                            gradients[(1e-5, name)][block_mask],
                            gradients[(1e-7, name)][block_mask],
                        ),
                    }
                )
        tight_rows = [
            r for r in report["gradient_records"] if r["forward_tolerance"] == 1e-7
        ]
        can_step = len(tight_rows) == len(CASES) and all(
            "full_parameters" in row for row in tight_rows
        )
        if can_step:
            max_norm = max(
                gradients[(1e-7, name)].norm().item() for name, _, _ in CASES
            )

            for relative_displacement in (1e-5, 1e-4):
                alpha = relative_displacement * parameter_norm / max(max_norm, 1e-30)
                for name, _, _ in CASES:
                    restore()
                    gradient = gradients[(1e-7, name)]
                    offset = 0
                    with torch.no_grad():
                        for _, parameter in parameters:
                            count = parameter.numel()
                            parameter.add_(
                                gradient[offset : offset + count].view_as(parameter),
                                alpha=-alpha,
                            )
                            offset += count

                    func, _ = capture()
                    step = {
                        "method": name,
                        "step_size": alpha,
                        "maximum_relative_displacement": relative_displacement,
                        "actual_relative_displacement": alpha
                        * gradient.norm().item()
                        / parameter_norm,
                    }
                    report["steps"].append(step)
                    next_states, step_info = solve(
                        func,
                        states[1e-7],
                        lambda row: notify(
                            {"step_forward": name, "alpha": alpha, **row}
                        ),
                    )
                    step["forward"] = step_info
                    step["status"] = (
                        "measured" if 1e-7 in next_states else "failed_forward_residual"
                    )
                    step["losses"] = {}

                    for tolerance, next_state in next_states.items():
                        with torch.no_grad():
                            for buffer_name, buffer in model.named_buffers():
                                buffer.copy_(original[buffer_name])
                        step["losses"][str(tolerance)] = loss_at(next_state)
                    if 1e-7 in next_states:
                        step["loss_change"] = step["losses"][str(1e-7)] - losses[1e-7]
                        step["predicted_loss_change"] = (
                            -alpha
                            * (gradients[(1e-7, "ift_reference")] * gradient)
                            .sum()
                            .item()
                        )
                        if "1e-05" in step["losses"] and 1e-5 in losses:
                            step["observed_forward_tolerance_uncertainty"] = abs(
                                step["losses"]["1e-05"] - step["losses"]["1e-07"]
                            ) + abs(losses[1e-5] - losses[1e-7])
                    del func, next_states
                    save()
        else:
            report["step_comparison_status"] = (
                "skipped_without_accepted_tight_forward_and_adjoint"
            )
        restore()
        report["model_state_restored"] = all(
            torch.equal(v, original[k]) for k, v in model.state_dict().items()
        )
        report["status"] = "completed_diagnostic"
        report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        save()
    except Exception:
        report["status"] = "failed_exception"
        report["exception"] = traceback.format_exc()
        save()
        raise


if __name__ == "__main__":
    main()
