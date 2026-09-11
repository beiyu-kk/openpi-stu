"""Check the tested STL dependency baseline and shared OpenPI imports."""

import importlib
from importlib import metadata
from pathlib import Path
import sys

from packaging.requirements import Requirement


def main():
    failures = []
    print(f"Python: {sys.executable} ({sys.version.split()[0]})")
    requirements = Path(__file__).with_name("requirements-openpi.txt")
    for line in requirements.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        requirement = Requirement(line)
        try:
            installed = metadata.version(requirement.name)
            matches = requirement.specifier.contains(installed)
        except metadata.PackageNotFoundError:
            installed, matches = "missing", False
        print(
            f"{'OK' if matches else 'MISMATCH'} {requirement.name}: {installed}; expected {requirement.specifier}"
        )
        if not matches:
            failures.append(requirement.name)
    for module in (
        "openpi.models.stl.locate_anything.modeling_locateanything",
        "openpi.models.stl.locate_anything.processing_locateanything",
        "openpi.models.pi0",
        "openpi.models_pytorch.pi0_pytorch",
    ):
        try:
            importlib.import_module(module)
            print(f"OK import {module}")
        except Exception as exc:
            print(f"FAIL import {module}: {type(exc).__name__}: {exc}")
            failures.append(module)
    if failures:
        raise SystemExit(1)
    from openpi.models.pi0_config import Pi0Config
    import torch

    print(f"Pi05 config: {Pi0Config(pi05=True).model_type.value}")
    print(
        f"CUDA available: {torch.cuda.is_available()}; PyTorch CUDA: {torch.version.cuda}"
    )
    print(
        "STL baseline and shared imports passed. This does not run a Pi05 training step."
    )


if __name__ == "__main__":
    main()
