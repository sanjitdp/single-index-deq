import json
from pathlib import Path
import torch
from model import make_model, state_digest


def main():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    original = make_model()
    torch.manual_seed(0)
    adapted = make_model(initial_state_gain=0.1)
    before, after = original.state_dict(), adapted.state_dict()
    changed = [name for name in before if not torch.equal(before[name], after[name])]
    expected = [
        f"full_stage.post_fuse_layers.{branch}.gnorm.weight" for branch in [0, 1]
    ]
    assert sorted(changed) == sorted(expected)
    for name in expected:
        assert torch.equal(after[name], 0.1 * before[name])
        assert dict(adapted.named_parameters())[name].requires_grad
    hashes = []
    config = json.loads((Path(__file__).parent / "configs.json").read_text())
    for case in config["configs"]:
        torch.manual_seed(0)
        model = make_model(
            method=case["method"],
            terms=case["terms"],
            damping=case["damping"],
            initial_state_gain=0.1,
        )
        hashes.append(state_digest(model))
    assert len(set(hashes)) == 1
    torch.manual_seed(0)
    offset = make_model(initial_branch_gain=0.1)
    offset_changed = [
        name
        for name, value in before.items()
        if not torch.equal(value, offset.state_dict()[name])
    ]
    expected_offset = [
        f"full_stage.branches.{branch}.blocks.0.gn3.{parameter}"
        for branch in [0, 1]
        for parameter in ["weight", "bias"]
    ]
    assert sorted(offset_changed) == sorted(expected_offset)
    assert all(
        dict(offset.named_parameters())[name].requires_grad for name in offset_changed
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "changed_tensors": changed,
                "offset_changed_tensors": offset_changed,
                "five_paired_state_hashes": hashes,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
