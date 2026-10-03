# Gradient descent dynamics for deep equilibrium models

Code and saved results for Appendix A. Run the commands below from this directory
with Python 3.11.

## Setup

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For CIFAR, also run `python -m pip install -r cifar/requirements.txt` with a
CUDA-compatible installation of PyTorch. We used PyTorch 2.6.0 and TorchVision
0.21.0. Install Computer Modern Unicode fonts to match the paper's plots.

## Files

| Directory or script | Contents |
| --- | --- |
| `synthetic/legacy_figures.py` | Scalar linear and sigmoid experiments, Appendix A.1–A.2 |
| `synthetic/approximate.py` | Linear counterexample and nonlinear backward approximations, Appendix A.3 |
| `synthetic/nonlinearity.py` | Weak-nonlinearity experiment, Appendix A.4 |
| `cifar/run.py` | Five single-seed CIFAR-10 training runs, Appendix A.5 |
| `cifar/diagnostics/shared_equilibrium.py` | Matched single-update comparisons, Appendix A.5 |
| `plot.py` | Paper-sized figures from saved results |
| `results/` | Recorded trajectories and summaries; no datasets or model checkpoints |

## Synthetic experiments and figures

To plot the saved results without training:

```sh
python run_synthetic.py --plot-only
```

To rerun the synthetic experiments in parallel and regenerate the figures:

```sh
python run_synthetic.py --overwrite-results
```

Figures are saved to `images/`. Rerunning replaces the saved synthetic results;
copy `results/` first if you want to keep them. The individual scripts also write
to this directory.

Figure 3 uses seed 6, chosen from seeds 0–19 for visual clarity; see
`synthetic/select_linear_illustration.py`. The sigmoid runs use 40,000 updates.
Plots show shorter portions of the trajectories, as described in the paper.

## CIFAR-10

Run indices 0–4 for JFB, two terms, three terms, damped two terms, and the
five-step phantom baseline, respectively. Each method uses one seed and the
settings in `cifar/configs.json`.

```sh
python cifar/run.py --index 0 --data data --output runs/cifar --download
```

Training defaults to CUDA; `--device cpu` and `--device mps` are also supported.
Use `--dry-run` to print the command or `--resume` to resume an interrupted run.
Results may vary across hardware.

These runs use a fixed forward-solver budget. To compare individual updates
using more accurate forward solves, first train the JFB and two-term models, then run:

```sh
python cifar/diagnostics/shared_equilibrium.py --index 0 --data data \
  --checkpoints runs/cifar --output runs/local/task-0.json \
  --adjoint-restart 100 --adjoint-iterations 2000
```

Repeat for indices 0–5, changing the output filename each time. This requires
CUDA and compares three batches for each checkpoint without changing the saved
models. Only load checkpoints you trust.

Saved reports are in `results/local/`. Tasks 0–4 passed the required tolerances;
task 5 failed the forward solve. The command above uses the larger adjoint budget
from the retries; the original task 0 used a smaller budget.

## Tests

```sh
python check.py
python check.py --cifar
```

The tests check gradients, forward solvers, backward approximations, and model
initialization on small inputs. They do not download CIFAR or run full training.

## Third-party code

We use [TorchDEQ](https://github.com/locuslab/torchdeq), commit
`4f6bd5fa66dd991cad74fcc847c88061764cf8db`. The required library and small MDEQ
model are included in `cifar/third_party/torchdeq`, with their MIT license and
upstream notices.
