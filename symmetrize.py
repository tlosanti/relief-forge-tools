#!/usr/bin/env python3
"""Compatibility wrapper. The implementation moved to ``cli/symmetrize.py``.

The ``--keep`` option now takes ``negative`` or ``positive`` rather than
``-x``/``+x``; the old spellings are translated here.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "cli"))

from symmetrize import main  # type: ignore[import-not-found]  # noqa: E402

_LEGACY_KEEP = {"-x": "negative", "+x": "positive"}


def _translate(argv: list[str]) -> list[str]:
    """Map legacy ``--keep -x`` / ``--keep +x`` spellings onto the new values."""
    out: list[str] = []
    for index, token in enumerate(argv):
        if index > 0 and argv[index - 1] == "--keep" and token in _LEGACY_KEEP:
            out.append(_LEGACY_KEEP[token])
        else:
            out.append(token)
    return out


if __name__ == "__main__":
    sys.exit(main(_translate(sys.argv[1:])))
