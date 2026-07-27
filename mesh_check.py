#!/usr/bin/env python3
"""Compatibility wrapper. The implementation moved to ``cli/mesh_check.py``.

Retained so that existing scripts and documented commands continue to work
after the package restructure. New usage should prefer::

    python cli/mesh_check.py piece.stl
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "cli"))

from mesh_check import main  # type: ignore[import-not-found]  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
