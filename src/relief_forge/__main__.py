"""Console-script entry points.

Thin adapters so that installed commands and the ``cli/`` scripts share one
implementation. Importing this module does not import PyTorch.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _cli_directory() -> Path:
    """Locate the ``cli/`` directory in a source checkout or installed tree."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "cli"
        if (candidate / "mesh_check.py").is_file():
            return candidate
    raise ModuleNotFoundError("cli/ directory not found alongside the package")


def _load(module_name: str):  # type: ignore[no-untyped-def]
    sys.path.insert(0, str(_cli_directory()))
    return __import__(module_name)


def mesh_check() -> int:
    """Entry point for the ``mesh-check`` command."""
    return int(_load("mesh_check").main())


def symmetrize() -> int:
    """Entry point for the ``symmetrize`` command."""
    return int(_load("symmetrize").main())


def generate() -> int:
    """Entry point for the ``relief-generate`` command."""
    return int(_load("generate").main())
