#!/usr/bin/env python3
"""Reconstruct a mesh from one half of itself by planar reflection.

Appropriate only when bilateral symmetry is a known property of the object.
Where symmetry holds, the operation replaces one half's reconstruction with the
other's; where it does not, it discards real asymmetric geometry.

The result is validated after the operation and the command exits non-zero if
it fails, so a symmetry pass cannot silently produce a non-manifold mesh.

Exit codes
----------
0
    Result passes validation.
1
    Result fails validation.
2
    Usage or input error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # type: ignore[import-not-found]
    EXIT_CHECKS_FAILED,
    EXIT_OK,
    EXIT_USAGE_ERROR,
    ensure_package_on_path,
    format_report,
)

ensure_package_on_path()

from relief_forge import ReliefForgeError, analyse_mesh, read_stl, write_stl  # noqa: E402
from relief_forge.symmetry import AXIS_INDEX, plane_position, symmetrize  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="symmetrize",
        description="Mirror one half of a mesh onto itself across a plane.",
    )
    parser.add_argument("stl", type=Path, help="input STL file")
    parser.add_argument("-o", "--out", type=Path, required=True, help="output STL file")
    parser.add_argument(
        "--axis",
        choices=sorted(AXIS_INDEX),
        default="x",
        help="mirror-plane normal (default: x, left/right)",
    )
    parser.add_argument(
        "--at",
        type=float,
        default=None,
        help="plane position along the axis (default: bounding-box midpoint)",
    )
    parser.add_argument(
        "--keep",
        choices=["negative", "positive"],
        default="negative",
        help="which half is retained (default: negative)",
    )
    parser.add_argument(
        "--expected-components",
        type=int,
        default=1,
        help="connected-component count required to pass (default: 1)",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        triangles = read_stl(args.stl)
        position = (
            plane_position(triangles, args.axis) if args.at is None else args.at
        )
        result = symmetrize(
            triangles,
            axis=args.axis,
            position=position,
            keep_positive=args.keep == "positive",
        )
        write_stl(args.out, result)
        report = analyse_mesh(args.out, expected_components=args.expected_components)
    except (ReliefForgeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE_ERROR

    if args.json:
        print(
            json.dumps(
                {
                    "input": str(args.stl),
                    "output": str(args.out),
                    "axis": args.axis,
                    "plane_position": position,
                    "kept_half": args.keep,
                    "triangles_in": len(triangles),
                    "triangles_out": len(result),
                    "report": report.to_dict(),
                },
                indent=2,
            )
        )
    else:
        default_note = " (bounding-box midpoint)" if args.at is None else ""
        print(f"plane: {args.axis} = {position:.6g}{default_note}")
        print(f"kept:  {args.keep} half")
        print(f"{len(triangles):,} triangles in -> {len(result):,} out\n")
        print(format_report(report))
        if not report.passes_checks:
            print(
                "\n  Validation failed. If the input was a closed manifold, the "
                "usual\n  cause is a plane grazing coplanar geometry; adjust --at "
                "slightly."
            )

    return EXIT_OK if report.passes_checks else EXIT_CHECKS_FAILED


if __name__ == "__main__":
    sys.exit(main())
