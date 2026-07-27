#!/usr/bin/env python3
"""Validate STL topology and report physical measurements.

Classifies a mesh as passing this repository's checks when it has no boundary
edges, no non-manifold edges, no degenerate edges, consistent local winding,
positive signed volume and the expected connected-component count.

Passing does not certify manufacturability. It establishes that the mesh is a
closed, coherently oriented solid, and says nothing about wall thickness,
overhangs, printer tolerance, casting shrinkage or trapped volumes.

Exit codes
----------
0
    All requested meshes passed.
1
    At least one mesh failed a check.
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

from relief_forge import ReliefForgeError, analyse_mesh  # noqa: E402
from relief_forge.measurements import MeshReport  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mesh-check",
        description="Validate STL topology and report physical measurements.",
    )
    parser.add_argument("stl", type=Path, help="STL file to analyse")
    parser.add_argument(
        "--compare",
        type=Path,
        metavar="OTHER.stl",
        help="second mesh to measure against the first",
    )
    parser.add_argument(
        "--expected-components",
        type=int,
        default=1,
        help="connected-component count required to pass (default: 1)",
    )
    parser.add_argument(
        "--weld-tolerance",
        type=float,
        default=1e-6,
        help="vertex welding tolerance as a fraction of the bounding-box "
        "diagonal (default: 1e-6)",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )
    return parser


def print_comparison(first: MeshReport, second: MeshReport) -> None:
    """Print a side-by-side table of two meshes."""
    print(f"\ncomparison: {first.source} (A) vs {second.source} (B)")
    print(f"  {'':<22}{'A':>14}{'B':>14}{'B vs A':>11}")

    rows: list[tuple[str, float, float]] = [
        ("triangles", float(first.triangles), float(second.triangles)),
        ("components", float(first.topology.components), float(second.topology.components)),
        ("bbox x (mm)", first.bbox_mm[0], second.bbox_mm[0]),
        ("bbox y (mm)", first.bbox_mm[1], second.bbox_mm[1]),
        ("bbox z (mm)", first.bbox_mm[2], second.bbox_mm[2]),
        ("volume (cm3)", first.volume_cm3, second.volume_cm3),
        ("area (cm2)", first.surface_area_mm2 / 100.0, second.surface_area_mm2 / 100.0),
        ("mean dihedral (deg)", first.mean_dihedral_deg, second.mean_dihedral_deg),
        (
            "silver mass (g)",
            first.material_masses_g()["silver_925"],
            second.material_masses_g()["silver_925"],
        ),
    ]

    for label, a, b in rows:
        delta = f"{(b - a) / abs(a) * 100:+.1f}%" if a else "-"
        print(f"  {label:<22}{a:>14.3f}{b:>14.3f}{delta:>11}")

    print(
        "\n  Mean dihedral angle is a detail-retention proxy for closely related\n"
        "  meshes only. Compare at identical scale and comparable triangle density."
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        first = analyse_mesh(
            args.stl,
            expected_components=args.expected_components,
            relative_tolerance=args.weld_tolerance,
        )
        second = (
            analyse_mesh(
                args.compare,
                expected_components=args.expected_components,
                relative_tolerance=args.weld_tolerance,
            )
            if args.compare
            else None
        )
    except (ReliefForgeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE_ERROR

    if args.json:
        payload = first.to_dict() if second is None else {
            "a": first.to_dict(),
            "b": second.to_dict(),
        }
        print(json.dumps(payload, indent=2))
    else:
        print(format_report(first))
        if second is not None:
            print()
            print(format_report(second))
            print_comparison(first, second)

    passed = first.passes_checks and (second.passes_checks if second else True)
    return EXIT_OK if passed else EXIT_CHECKS_FAILED


if __name__ == "__main__":
    sys.exit(main())
