import argparse
import ast
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tokenize

ROOT = Path(__file__).resolve().parent
for key in ["OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS"]:
    os.environ[key] = "1"


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "synthetic" / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cifar", action="store_true")
    args = parser.parse_args()
    for path in ROOT.rglob("*.py"):
        if any(part in path.parts for part in ["third_party", ".venv", "venv", "tmp"]):
            continue
        source = path.read_text()
        tree = ast.parse(source)
        assert not any(
            t.type == tokenize.COMMENT
            for t in tokenize.generate_tokens(io.StringIO(source).readline)
        ), path
        for node in ast.walk(tree):
            if isinstance(
                node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                assert ast.get_docstring(node) is None, path
    print("Source syntax and comment checks passed.")
    print("Approximate backpropagation:", load("approximate").checks())
    nonlinear = load("nonlinearity")
    x, _, xi, theta = nonlinear.make_data(1729, 10, 128, 128)
    for epsilon in [0.5, 0.25, 0.125]:
        print("Weak nonlinearity:", epsilon, nonlinear.validate(x, xi, theta, epsilon))
    print("Scalar models:", load("legacy_figures").validate())
    if args.cifar:
        for name in [
            "test_backward.py",
            "test_accurate_forward.py",
            "test_recipe.py",
            "test_initialization.py",
            "diagnostics/test_shared_equilibrium.py",
        ]:
            subprocess.run(
                [sys.executable, str(ROOT / "cifar" / name)], cwd=ROOT, check=True
            )


if __name__ == "__main__":
    main()
