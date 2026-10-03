import json
import numpy as np
from legacy_figures import OUT, run


def main():
    candidates = []
    for seed in range(20):
        initial = np.random.default_rng(seed).normal(0, 0.1, 2) + [0, 1]
        result = run("linear", initial, 200, 0.01)
        increments = np.linalg.norm(np.diff(result["theta"], axis=0), axis=1)
        candidates.append(
            {
                "seed": seed,
                "initial_theta": initial.tolist(),
                "max_step": float(increments.max()),
                "path_length": float(increments.sum()),
                "final_risk": float(result["risk"][-1]),
            }
        )

    selected = candidates[6]
    result = run("linear", selected["initial_theta"], 200, 0.01)
    assert np.all(np.isfinite(result["theta"]))
    assert np.all(result["theta"][:, 1] > 1)
    distances = np.linalg.norm(result["theta"] - [0, 1], axis=1)
    assert np.all(np.diff(distances) >= -1e-12)
    assert result["risk"][-1] < 1e-8
    np.savez_compressed(OUT / "linear-near-singular-illustration.npz", **result)
    report = {
        "selected_initialization_seed": 6,
        "data_seed": 42,
        "selection": "Illustrative trajectory chosen from seeds 0--19 for visual clarity.",
        "original_result": "linear-near-singular.npz (unchanged)",
        "candidates": candidates,
    }
    (OUT / "illustration-selection.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
