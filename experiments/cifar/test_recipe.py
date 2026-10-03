import argparse
import json
from pathlib import Path
import torch
from model import make_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(31)
    reference = make_model(packaged_forward=True).to(args.device).train()
    torch.manual_seed(31)
    adapted = make_model(method="phantom", terms=5, damping=0.5).to(args.device).train()
    images = torch.randn(2, 3, 32, 32, device=args.device)
    results = []
    for model in [reference, adapted]:
        logits = model(images)[-1]
        logits.square().mean().backward()
        results.append(
            (
                logits.detach().cpu(),
                torch.cat(
                    [
                        p.grad.detach().reshape(-1).cpu()
                        for p in model.parameters()
                        if p.grad is not None
                    ]
                ),
            )
        )
    logit_error = (results[0][0] - results[1][0]).abs().max().item()
    gradient_error = (results[0][1] - results[1][1]).abs().max().item()
    report = {
        "device": args.device,
        "max_logit_difference": logit_error,
        "max_gradient_difference": gradient_error,
        "status": "passed",
    }
    assert logit_error < 1e-5 and gradient_error < 1e-5, report
    print(json.dumps(report, indent=2))
    dest = Path(__file__).resolve().parents[1] / "results/cifar/recipe_test.json"
    dest.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
