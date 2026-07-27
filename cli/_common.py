"""Shared CLI helpers: exit codes, path setup and report rendering.

Presentation lives here and in the sibling CLI modules. The geometry package
itself performs no printing and raises exceptions rather than exiting, so it
remains usable as a library.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

#: Conventional exit codes used by every command in this package.
EXIT_OK: Final[int] = 0
EXIT_CHECKS_FAILED: Final[int] = 1
EXIT_USAGE_ERROR: Final[int] = 2


def ensure_package_on_path() -> None:
    """Add ``src/`` to ``sys.path`` when running from a source checkout.

    Allows the CLI to run directly from a clone without installation, while
    remaining a no-op once the package is installed.
    """
    try:
        import relief_forge  # noqa: F401

        return
    except ImportError:
        pass

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "src"
        if (candidate / "relief_forge" / "__init__.py").is_file():
            sys.path.insert(0, str(candidate))
            return


def format_report(report: MeshReport) -> str:  # noqa: F821 - imported lazily
    """Render a :class:`relief_forge.measurements.MeshReport` as plain text."""
    topology = report.topology
    verdict = "PASS" if report.passes_checks else "FAIL"
    width, depth, height = report.bbox_mm

    lines = [
        f"{report.source} - {verdict}",
        f"  triangles           {report.triangles:,} ({report.vertices:,} welded vertices)",
        f"  bounding box        {width:.3f} x {depth:.3f} x {height:.3f} mm",
        f"  volume              {report.volume_cm3:.4f} cm3",
        f"  surface area        {report.surface_area_mm2 / 100.0:.3f} cm2",
        f"  mean dihedral       {report.mean_dihedral_deg:.2f} deg (detail-retention proxy)",
        f"  components          {topology.components} (expected {report.expected_components})",
        f"  closed manifold     {'yes' if topology.is_closed_manifold else 'no'}",
    ]

    if not topology.is_closed_manifold:
        labels = {
            "boundary_edges": "boundary edges (holes)",
            "nonmanifold_edges": "non-manifold edges (more than two faces)",
            "inconsistent_edges": "inconsistently wound edges",
            "degenerate_edges": "degenerate edges (zero length)",
        }
        for key, count in topology.defects().items():
            lines.append(f"      {count:,} {labels[key]}")

    lines.append(
        f"  outward wound       {'yes' if report.is_outward_wound else 'no (normals inverted)'}"
    )

    masses = report.material_masses_g()
    lines.append(
        "  estimated mass      "
        f"silver {masses['silver_925']:.2f} g | bronze {masses['bronze']:.2f} g | "
        f"18k gold {masses['gold_18k']:.2f} g | resin {masses['resin']:.2f} g"
    )
    return "\n".join(lines)
