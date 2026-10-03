import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
import random
import time
import numpy as np
import torch
from model import make_model, COMMIT, HERE, default_device, state_digest
from torchvision import datasets, transforms
from accurate_forward import ForwardSolveError


def evaluate(model, loader, device):
    model.eval()
    loss_sum = correct = count = 0
    max_forward = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            max_forward = max(
                max_forward,
                model.deq.stats.get(
                    "forward_max_relative_residual",
                    model.deq.stats["forward_relative_residual"],
                ),
            )
            loss_sum += torch.nn.functional.cross_entropy(
                logits, labels, reduction="sum"
            ).item()
            correct += (logits.argmax(1) == labels).sum().item()
            count += len(labels)
    return {
        "loss": loss_sum / count,
        "accuracy": correct / count,
        "samples": count,
        "max_forward_relative_residual": max_forward,
        "residual_aggregation": (
            "max_example"
            if model.deq.forward_residual_limit is not None
            else "max_batch_mean"
        ),
    }


def run(args):
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable. Choose an available device with --device.")
    device = torch.device(args.device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    if args.full_precision:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    args.output.mkdir(parents=True, exist_ok=args.resume)
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata.update(
        upstream_commit=COMMIT,
        torch=torch.__version__,
        python=platform.python_version(),
        numpy=np.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        deterministic_algorithms=False,
        determinism_note="Paired RNG seeds, no augmentation, single data-loader process; MPS kernels are not promised bitwise deterministic.",
    )
    if args.resume:
        previous = json.loads((args.output / "config.json").read_text())
        for key in [
            "dataset",
            "phase",
            "method",
            "terms",
            "damping",
            "seed",
            "lr",
            "epochs",
            "batch_size",
            "device",
            "forward_iterations",
            "backward_iterations",
            "ift_residual_limit",
        ]:
            if previous[key] != metadata[key]:
                raise ValueError(f"Cannot change {key} on resume")
        for key, default in [("forward_tolerance", 1e-3), ("backward_tolerance", 1e-6)]:
            if previous.get(key, default) != metadata[key]:
                raise ValueError(f"Cannot change {key} on resume")
        if previous.get("forward_residual_limit") != args.forward_residual_limit:
            raise ValueError("Cannot change forward_residual_limit on resume")
        if previous.get("initial_state_gain", 1.0) != args.initial_state_gain:
            raise ValueError("Cannot change initial_state_gain on resume")
        if previous.get("initial_branch_gain") != args.initial_branch_gain:
            raise ValueError("Cannot change initial_branch_gain on resume")
        for key, default in [("forward_policy", "standard"), ("full_precision", False)]:
            if previous.get(key, default) != getattr(args, key):
                raise ValueError(f"Cannot change {key} on resume")
    else:
        (args.output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    cls = datasets.CIFAR10 if args.dataset == "cifar10" else datasets.CIFAR100
    data = cls(str(args.data), train=True, download=args.download, transform=transform)
    permutation = np.random.default_rng(314159).permutation(len(data))
    train_indices, validation_indices = permutation[:45000], permutation[45000:]
    if args.smoke_updates:
        train_indices, validation_indices = (
            train_indices[:256],
            validation_indices[:128],
        )
    metadata["split_sha256"] = hashlib.sha256(permutation.tobytes()).hexdigest()
    if not args.resume:
        (args.output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    generator = torch.Generator().manual_seed(args.seed + 10000)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(data, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    train_eval_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(data, train_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(data, validation_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = make_model(
        classes=10 if args.dataset == "cifar10" else 100,
        method=args.method,
        terms=args.terms,
        damping=args.damping,
        forward_iterations=args.forward_iterations,
        forward_tolerance=args.forward_tolerance,
        backward_iterations=args.backward_iterations,
        backward_tolerance=args.backward_tolerance,
        forward_residual_limit=args.forward_residual_limit,
        initial_state_gain=args.initial_state_gain,
        initial_branch_gain=args.initial_branch_gain,
        forward_policy=args.forward_policy,
    ).to(device)
    metadata["initial_state_sha256"] = state_digest(model)
    if not args.resume:
        (args.output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001
    )
    total_updates = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, total_updates, eta_min=1e-6
    )
    history = []
    started = time.perf_counter()
    train_seconds = 0.0
    updates = 0
    begin_epoch = 0
    elapsed_before_resume = 0.0
    if args.resume:

        checkpoint = torch.load(
            args.output / "checkpoint.pt", map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history = checkpoint["history"]
        begin_epoch = checkpoint["epoch"]
        updates = checkpoint["updates"]
        train_seconds = checkpoint["training_seconds"]
        elapsed_before_resume = checkpoint["elapsed_seconds"]
        generator.set_state(checkpoint["loader_rng"].cpu())
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([v.cpu() for v in checkpoint["device_rng"]])
        elif device.type == "mps":
            torch.mps.set_rng_state(checkpoint["device_rng"].cpu())
    for epoch in range(begin_epoch, args.epochs):
        model.train()
        loss_sum = count = 0
        max_forward_residual = max_backward_residual = 0.0
        max_example_residual = forward_calls = forward_retries = 0
        epoch_start = time.perf_counter()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            try:
                logits = model(images)[-1]
            except ForwardSolveError as error:
                error.stats.update(epoch=epoch + 1, completed_updates=updates)
                raise
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            adjoint_residual = model.deq.stats.get("backward_relative_residual", 0.0)
            if args.method == "ift" and (
                not math.isfinite(adjoint_residual)
                or adjoint_residual > args.ift_residual_limit
            ):
                failure = {
                    "status": "failed_ift_residual",
                    "epoch": epoch,
                    "updates": updates,
                    "adjoint_relative_residual": adjoint_residual,
                    "config": metadata,
                }
                (args.output / "result.json").write_text(
                    json.dumps(failure, indent=2) + "\n"
                )
                raise RuntimeError(
                    "Numerical IFT residual too large; update NOT applied. Inspect solver convergence before a full sweep."
                )
            if not torch.isfinite(loss) or not all(
                p.grad is None or torch.isfinite(p.grad).all()
                for p in model.parameters()
            ):
                failure = {
                    "status": "failed_nonfinite",
                    "epoch": epoch,
                    "updates": updates,
                    **metadata,
                }
                (args.output / "result.json").write_text(
                    json.dumps(failure, indent=2) + "\n"
                )
                raise FloatingPointError(
                    "Nonfinite loss/gradient; failure recorded, no clipping or silent retry."
                )
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item() * len(labels)
            count += len(labels)
            updates += 1
            if updates % 25 == 0:
                progress = {
                    "status": "training",
                    "epoch": epoch + 1,
                    "updates": updates,
                    "loss": loss.item(),
                    "forward_stats": model.deq.stats,
                    "elapsed_seconds": elapsed_before_resume
                    + time.perf_counter()
                    - started,
                }
                temporary = args.output / "progress.tmp"
                temporary.write_text(json.dumps(progress, indent=2) + "\n")
                temporary.replace(args.output / "progress.json")
                print(json.dumps(progress), flush=True)
            max_forward_residual = max(
                max_forward_residual, model.deq.stats["forward_relative_residual"]
            )
            max_example_residual = max(
                max_example_residual,
                model.deq.stats.get("forward_max_relative_residual", 0.0),
            )
            forward_calls += model.deq.stats.get("forward_function_evaluations", 0)
            forward_retries += model.deq.stats.get("forward_retries", 0)
            max_backward_residual = max(
                max_backward_residual,
                model.deq.stats.get("backward_relative_residual", 0.0),
            )
            if args.smoke_updates and updates >= args.smoke_updates:
                break
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds += time.perf_counter() - epoch_start
        val = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch + 1,
            "updates": updates,
            "online_train_loss": loss_sum / count,
            "validation": val,
            "max_batch_mean_forward_relative_residual": max_forward_residual,
            "max_backward_relative_residual": max_backward_residual,
            "elapsed_seconds": elapsed_before_resume + time.perf_counter() - started,
        }
        if args.forward_residual_limit is not None:
            row.update(
                max_example_forward_relative_residual=max_example_residual,
                forward_function_evaluations=forward_calls,
                forward_retry_batches=forward_retries,
            )
        history.append(row)
        print(json.dumps(row), flush=True)
        (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1,
            "updates": updates,
            "history": history,
            "training_seconds": train_seconds,
            "elapsed_seconds": row["elapsed_seconds"],
            "loader_rng": generator.get_state(),
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "config": metadata,
        }
        if device.type == "cuda":
            checkpoint["device_rng"] = torch.cuda.get_rng_state_all()
        elif device.type == "mps":
            checkpoint["device_rng"] = torch.mps.get_rng_state()
        torch.save(checkpoint, args.output / "checkpoint.tmp")
        (args.output / "checkpoint.tmp").replace(args.output / "checkpoint.pt")
        if args.smoke_updates and updates >= args.smoke_updates:
            break
    result = {
        "status": "smoke_only" if args.smoke_updates else "completed",
        "updates": updates,
        "final_train": evaluate(model, train_eval_loader, device),
        "final_validation": history[-1]["validation"],
        "training_seconds": train_seconds,
        "config": metadata,
    }

    if args.phase == "final" and not args.smoke_updates:
        test = cls(
            str(args.data), train=False, download=args.download, transform=transform
        )
        result["final_test"] = evaluate(
            model,
            torch.utils.data.DataLoader(
                test, batch_size=args.batch_size, shuffle=False, num_workers=0
            ),
            device,
        )
    result["total_seconds"] = elapsed_before_resume + time.perf_counter() - started
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": metadata,
        },
        args.output / "final.pt",
    )
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--phase", choices=["tune", "final"], default="tune")
    parser.add_argument(
        "--method", choices=["neumann", "ift", "phantom"], default="phantom"
    )
    parser.add_argument("--terms", type=int, default=5)
    parser.add_argument("--damping", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--device", choices=["cuda", "mps", "cpu"], default=default_device()
    )
    parser.add_argument("--forward-iterations", type=int, default=22)
    parser.add_argument("--backward-iterations", type=int, default=50)
    parser.add_argument("--forward-tolerance", type=float, default=1e-3)
    parser.add_argument("--initial-state-gain", type=float, default=1.0)
    parser.add_argument("--initial-branch-gain", type=float, default=None)
    parser.add_argument(
        "--forward-policy", choices=["standard", "strong"], default="standard"
    )
    parser.add_argument(
        "--full-precision",
        action="store_true",
        help="Disable TF32 and convolution autotuning for residual fidelity",
    )
    parser.add_argument(
        "--forward-residual-limit",
        type=float,
        default=None,
        help="Enable accurate relative solves and reject every uncertified batch.",
    )
    parser.add_argument("--backward-tolerance", type=float, default=1e-6)
    parser.add_argument("--ift-residual-limit", type=float, default=1e-3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--data", type=Path, default=HERE / "data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--smoke-updates", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.forward_residual_limit is not None and not (
        0 < args.forward_tolerance <= args.forward_residual_limit
    ):
        parser.error("Require 0 < forward tolerance <= forward residual limit")
    try:
        run(args)
    except ForwardSolveError as error:
        failure = {
            "status": "failed_forward_residual",
            "diagnostics": error.stats,
            "note": "No update or evaluation accepted for the failing batch. Earlier epochs remain checkpointed.",
        }
        (args.output / "result.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
