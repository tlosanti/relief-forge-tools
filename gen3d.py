#!/usr/bin/env python3
"""Compatibility wrapper. The implementation moved to ``cli/generate.py``.

The ``--octree`` option was renamed ``--resolution`` and ``--faces`` became
``--max-faces``; both old spellings are translated here.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "cli"))

from generate import main  # type: ignore[import-not-found]  # noqa: E402

_RENAMED = {"--octree": "--resolution", "--faces": "--max-faces"}


def _translate(argv: list[str]) -> list[str]:
    """Rewrite renamed option flags, leaving their values untouched."""
    return [_RENAMED.get(token, token) for token in argv]


if __name__ == "__main__":
    sys.exit(main(_translate(sys.argv[1:])))
