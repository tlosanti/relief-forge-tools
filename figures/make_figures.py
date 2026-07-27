#!/usr/bin/env python3
"""Generate every technical figure in the repository.

All figures are produced from synthetic geometry with known analytical
properties, or from real example data under ``examples/``. Nothing is drawn by
hand and nothing is illustrative: each figure renders the same meshes the test
suite validates.

Style constraints: white background, thin black geometry edges, Okabe-Ito
categorical colours (colourblind-safe), equal axis scaling on all spatial
plots, labelled units, vector output. No Seaborn, no gradients, no shading.

    python figures/make_figures.py
    python figures/make_figures.py --formats pdf svg png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from conftest import (  # noqa: E402  # type: ignore[import-not-found]
    build_annulus,
    build_box,
    build_sphere,
)
from relief_forge import signed_volume, surface_area, weld  # noqa: E402
from relief_forge.measurements import (  # noqa: E402
    DENSITIES_G_PER_CM3,
    material_masses,
    mean_dihedral_angle,
)
from relief_forge.symmetry import clip_to_halfspace, symmetrize  # noqa: E402
from relief_forge.topology import analyse_topology, edge_incidence  # noqa: E402

OUTPUT_DIR = ROOT / "figures" / "generated"

# Okabe-Ito qualitative palette. Distinguishable under common colour-vision
# deficiencies and in greyscale print.
BLACK = "#000000"
ORANGE = "#E69F00"
SKY = "#56B4E9"
GREEN = "#009E73"
VERMILLION = "#D55E00"
BLUE = "#0072B2"

SURFACE_GREY = "#DDDDDD"
EDGE_WIDTH = 0.35

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "serif",
        "font.size": 8,
        "axes.linewidth": 0.6,
        "axes.grid": False,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)


# --------------------------------------------------------------------------
# Shared drawing helpers
# --------------------------------------------------------------------------

def set_equal_3d(ax, triangles: np.ndarray, margin: float = 0.05) -> None:
    """Apply identical spatial scaling to all three axes.

    Matplotlib's 3D axes do not support ``aspect='equal'`` reliably, so limits
    are computed around a common centre using one half-range for every axis.
    """
    points = triangles.reshape(-1, 3)
    centre = (points.max(axis=0) + points.min(axis=0)) / 2.0
    half = (points.max(axis=0) - points.min(axis=0)).max() / 2.0
    half *= 1.0 + margin
    for setter, value in zip(
        (ax.set_xlim, ax.set_ylim, ax.set_zlim), centre, strict=True
    ):
        setter(value - half, value + half)
    ax.set_box_aspect((1, 1, 1))


def draw_mesh(ax, triangles: np.ndarray, facecolor: str = SURFACE_GREY, alpha: float = 1.0) -> None:
    """Draw a triangle mesh as flat shaded polygons with thin black edges."""
    collection = Poly3DCollection(
        triangles,
        facecolors=facecolor,
        edgecolors=BLACK,
        linewidths=EDGE_WIDTH,
        alpha=alpha,
    )
    collection.set_zsort("average")
    ax.add_collection3d(collection)


def style_3d(ax, title: str, elev: float = 22.0, azim: float = -58.0) -> None:
    """Apply a consistent camera and minimal axis decoration."""
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=8, pad=2)
    ax.set_xlabel("x (mm)", labelpad=-6, fontsize=7)
    ax.set_ylabel("y (mm)", labelpad=-6, fontsize=7)
    ax.set_zlabel("z (mm)", labelpad=-6, fontsize=7)
    ax.tick_params(labelsize=5, pad=-2)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_edgecolor(BLACK)
        pane.pane.set_alpha(0.03)


def save(fig, stem: str, formats: list[str]) -> list[Path]:
    """Write a figure in every requested format and return the paths."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for extension in formats:
        path = OUTPUT_DIR / f"{stem}.{extension}"
        fig.savefig(path, dpi=300 if extension == "png" else None)
        written.append(path)
    plt.close(fig)
    return written


# --------------------------------------------------------------------------
# Figure 3: symmetry operation
# --------------------------------------------------------------------------

def figure_symmetry(formats: list[str]) -> list[Path]:
    """Three panels: asymmetric input, retained half with plane, mirrored result.

    Camera and axis limits are identical across panels so that the change in
    geometry is the only visible difference.
    """
    original = build_sphere(radius=8.0, subdivisions=3, offset=(2.0, 0.0, 0.0))
    half = clip_to_halfspace(original, "x", 0.0, keep_positive=False)
    result = symmetrize(original, "x", 0.0)

    fig = plt.figure(figsize=(7.4, 2.7))
    panels = [
        (original, "(a) input, asymmetric about $x=0$", None),
        (half, "(b) retained half, $x \\leq 0$", 0.0),
        (result, "(c) mirrored result", None),
    ]

    # A common frame for every panel, taken from the widest mesh.
    frame = np.concatenate([original, result])

    for index, (mesh, title, plane) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 3, index, projection="3d")
        draw_mesh(ax, mesh)
        if plane is not None:
            extent = 9.0
            corners = np.array(
                [
                    [plane, -extent, -extent],
                    [plane, extent, -extent],
                    [plane, extent, extent],
                    [plane, -extent, extent],
                ]
            )
            ax.add_collection3d(
                Poly3DCollection(
                    [corners], facecolors=VERMILLION, alpha=0.16, edgecolors=VERMILLION,
                    linewidths=0.6,
                )
            )
        set_equal_3d(ax, frame)
        style_3d(ax, title)

    volumes = [signed_volume(original), signed_volume(result)]
    fig.suptitle(
        f"Symmetry reconstruction. Input volume {volumes[0] / 1000:.3f} cm$^3$; "
        f"result {volumes[1] / 1000:.3f} cm$^3$. "
        "Valid only where bilateral symmetry is a known property of the object.",
        fontsize=7.5,
        y=0.04,
    )
    return save(fig, "fig_symmetry_operation", formats)


# --------------------------------------------------------------------------
# Figure 4: topology diagnostics
# --------------------------------------------------------------------------

def _classify_edges(triangles: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return welded vertices and edge index arrays grouped by classification."""
    welded = weld(triangles)
    unique, forward, reverse = edge_incidence(welded.faces)
    total = forward + reverse

    return welded.vertices, {
        "manifold": unique[(total == 2) & (forward == reverse)],
        "boundary": unique[total == 1],
        "nonmanifold": unique[total > 2],
        "inconsistent": unique[(total == 2) & (forward != reverse)],
    }


def figure_topology(formats: list[str]) -> list[Path]:
    """Four synthetic meshes, one per edge classification, with defects marked."""
    cube = build_box()

    missing_face = cube[:-2]
    flap = np.concatenate([cube, build_box(low=(0, 0, 10), high=(10, 10, 20))[:2]])
    reversed_tri = cube.copy()
    reversed_tri[-1] = reversed_tri[-1][::-1]

    cases = [
        (cube, "(a) valid closed manifold"),
        (missing_face, "(b) boundary edges"),
        (flap, "(c) non-manifold edge"),
        (reversed_tri, "(d) inconsistent winding"),
    ]
    colours = {
        "manifold": BLACK,
        "boundary": VERMILLION,
        "nonmanifold": BLUE,
        "inconsistent": ORANGE,
    }
    widths = {"manifold": 0.35, "boundary": 2.2, "nonmanifold": 2.2, "inconsistent": 2.2}

    fig = plt.figure(figsize=(7.4, 2.3))

    for index, (mesh, title) in enumerate(cases, start=1):
        ax = fig.add_subplot(1, 4, index, projection="3d")
        draw_mesh(ax, mesh, facecolor=SURFACE_GREY, alpha=0.55)

        vertices, groups = _classify_edges(mesh)
        for kind, edges in groups.items():
            if len(edges) == 0:
                continue
            segments = [vertices[[a, b]] for a, b in edges]
            ax.add_collection3d(
                Line3DCollection(
                    segments, colors=colours[kind], linewidths=widths[kind]
                )
            )

        welded = weld(mesh)
        report = analyse_topology(welded.faces, len(welded.vertices))
        counts = report.defects()
        subtitle = ", ".join(f"{v} {k.replace('_edges', '')}" for k, v in counts.items())
        set_equal_3d(ax, flap)
        style_3d(ax, f"{title}\n{subtitle or 'no defects'}")

    handles = [
        Line2D([], [], color=colours["manifold"], lw=1.2, label="manifold edge"),
        Line2D([], [], color=colours["boundary"], lw=2.2, label="boundary edge"),
        Line2D([], [], color=colours["nonmanifold"], lw=2.2, label="non-manifold edge"),
        Line2D([], [], color=colours["inconsistent"], lw=2.2, label="inconsistent winding"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    return save(fig, "fig_topology_diagnostics", formats)


# --------------------------------------------------------------------------
# Figure 5: validation comparison
# --------------------------------------------------------------------------

def figure_comparison(formats: list[str]) -> list[Path]:
    """Compare two tessellations of the same solid across measured quantities.

    Values are normalised to mesh A so that quantities with different units share
    an axis; absolute values are printed above each pair.
    """
    coarse = build_sphere(radius=15.0, subdivisions=2, offset=(0.0, 0.0, 0.0))
    fine = build_sphere(radius=15.0, subdivisions=4, offset=(0.0, 0.0, 0.0))

    def measure(mesh: np.ndarray) -> dict[str, float]:
        welded = weld(mesh)
        report = analyse_topology(welded.faces, len(welded.vertices))
        volume = signed_volume(mesh)
        return {
            "triangles": float(len(mesh)),
            "components": float(report.components),
            "volume\n(cm$^3$)": abs(volume) / 1000.0,
            "area\n(cm$^2$)": surface_area(mesh) / 100.0,
            "mean dihedral\n(deg)": mean_dihedral_angle(welded.vertices, welded.faces),
            "silver mass\n(g)": material_masses(volume)["silver_925"],
        }

    a, b = measure(coarse), measure(fine)
    labels = list(a)
    ratios = [b[k] / a[k] if a[k] else 0.0 for k in labels]

    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    positions = np.arange(len(labels))
    width = 0.36

    ax.bar(positions - width / 2, [1.0] * len(labels), width, color=SKY,
           edgecolor=BLACK, linewidth=0.5, label="A: 128 triangles")
    ax.bar(positions + width / 2, ratios, width, color=ORANGE,
           edgecolor=BLACK, linewidth=0.5, label="B: 2048 triangles")

    for index, key in enumerate(labels):
        ax.text(index - width / 2, 1.02, f"{a[key]:g}", ha="center", va="bottom", fontsize=6)
        ax.text(index + width / 2, ratios[index] + 0.02, f"{b[key]:g}",
                ha="center", va="bottom", fontsize=6)

    ax.axhline(1.0, color=BLACK, lw=0.5, ls=":")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("value relative to mesh A")
    ax.set_ylim(0, max([*ratios, 1.0]) * 1.30)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title(
        "Measured comparison of two tessellations of one sphere (radius 15 mm)",
        fontsize=8,
    )
    fig.text(
        0.5, -0.13,
        "Mean dihedral angle is a limited detail-retention proxy for closely related meshes at "
        "identical scale.\nIt is not a perceptual quality metric and is not comparable across "
        "unrelated meshes or tessellation densities.",
        ha="center", fontsize=6.5,
    )
    return save(fig, "fig_validation_comparison", formats)


# --------------------------------------------------------------------------
# Figure 6: material mass estimates
# --------------------------------------------------------------------------

def figure_material_mass(formats: list[str]) -> list[Path]:
    """Estimated cast mass per material for one measured reference solid."""
    mesh = build_annulus()
    volume_mm3 = abs(signed_volume(mesh))
    masses = material_masses(volume_mm3)

    display = {
        "silver_925": "sterling silver\n(925)",
        "bronze": "bronze",
        "gold_18k": "18k gold",
        "resin": "cast resin",
    }
    keys = ["silver_925", "bronze", "gold_18k", "resin"]
    values = [masses[k] for k in keys]

    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    bars = ax.bar(
        [display[k] for k in keys], values,
        color=[SKY, GREEN, ORANGE, BLUE], edgecolor=BLACK, linewidth=0.5, width=0.62,
    )
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.02,
                f"{value:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("estimated mass (g)")
    ax.set_ylim(0, max(values) * 1.18)
    ax.tick_params(labelsize=7)
    ax.set_title("Estimated cast mass by material", fontsize=8)

    densities = ", ".join(f"{display[k].splitlines()[0]} {DENSITIES_G_PER_CM3[k]}" for k in keys)
    fig.text(
        0.5, -0.16,
        f"Reference solid: annulus, outer 20 mm, inner 8 mm, height 3 mm; measured volume "
        f"{volume_mm3 / 1000:.4f} cm$^3$.\nAssumes a fully dense solid; excludes sprue, "
        f"flashing and shrinkage. Densities (g/cm$^3$): {densities}.",
        ha="center", fontsize=6.5,
    )
    return save(fig, "fig_material_mass", formats)


# --------------------------------------------------------------------------
# Figures requiring real example data
# --------------------------------------------------------------------------

def figure_image_to_mesh(formats: list[str]) -> list[Path]:
    """Source photograph beside front, side and isometric views of the result.

    Requires a real example: a photograph in ``examples/input/`` and the
    corresponding meshes in ``examples/output/``. No substitute is drawn, since
    an invented render would misrepresent what the pipeline produces.
    """
    photo = sorted((ROOT / "examples" / "input").glob("*.[jp][pn]g"))
    raw = ROOT / "examples" / "output" / "raw.stl"
    processed = ROOT / "examples" / "output" / "processed.stl"

    if not photo or not raw.is_file() or not processed.is_file():
        print(
            "  skipped fig_image_to_mesh: needs examples/input/<photo> plus\n"
            "    examples/output/raw.stl and examples/output/processed.stl"
        )
        return []

    from matplotlib.image import imread

    from relief_forge import read_stl

    raw_mesh, processed_mesh = read_stl(raw), read_stl(processed)
    fig = plt.figure(figsize=(7.4, 2.5))

    ax = fig.add_subplot(1, 4, 1)
    ax.imshow(imread(photo[0]))
    ax.set_title("(a) source photograph", fontsize=8)
    ax.axis("off")

    views = [
        (raw_mesh, "(b) generated surface", 0.0, -90.0),
        (processed_mesh, "(c) processed, front", 0.0, -90.0),
        (processed_mesh, "(d) processed, isometric", 22.0, -58.0),
    ]
    frame = np.concatenate([raw_mesh, processed_mesh])
    for index, (mesh, title, elev, azim) in enumerate(views, start=2):
        ax = fig.add_subplot(1, 4, index, projection="3d")
        draw_mesh(ax, mesh)
        set_equal_3d(ax, frame)
        style_3d(ax, title, elev=elev, azim=azim)

    return save(fig, "fig_image_to_mesh", formats)


def figure_multiview(formats: list[str]) -> list[Path]:
    """Geometry from one, two and four input views, with measured statistics.

    Requires ``examples/output/views_{1,2,4}.stl`` produced by real generation
    runs. Skipped otherwise: the comparison is only meaningful with measured
    output.
    """
    directory = ROOT / "examples" / "output"
    paths = [directory / f"views_{n}.stl" for n in (1, 2, 4)]
    if not all(path.is_file() for path in paths):
        print(
            "  skipped fig_multiview: needs examples/output/views_1.stl, "
            "views_2.stl, views_4.stl\n    from real generation runs"
        )
        return []

    from relief_forge import read_stl

    meshes = [read_stl(path) for path in paths]
    frame = np.concatenate(meshes)

    fig = plt.figure(figsize=(7.4, 2.7))
    for index, (mesh, count) in enumerate(zip(meshes, (1, 2, 4), strict=True), start=1):
        ax = fig.add_subplot(1, 3, index, projection="3d")
        draw_mesh(ax, mesh)
        set_equal_3d(ax, frame)
        welded = weld(mesh)
        report = analyse_topology(welded.faces, len(welded.vertices))
        style_3d(
            ax,
            f"({'abc'[index - 1]}) {count} view{'s' if count > 1 else ''}\n"
            f"{len(mesh):,} tri, {abs(signed_volume(mesh)) / 1000:.3f} cm$^3$, "
            f"{report.components} comp.",
        )
    return save(fig, "fig_multiview", formats)


# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Generate repository figures.")
    parser.add_argument(
        "--formats", nargs="+", default=["pdf", "svg"], choices=["pdf", "svg", "png"]
    )
    args = parser.parse_args()

    builders = [
        ("symmetry operation", figure_symmetry),
        ("topology diagnostics", figure_topology),
        ("validation comparison", figure_comparison),
        ("material mass", figure_material_mass),
        ("image to mesh", figure_image_to_mesh),
        ("multi-view", figure_multiview),
    ]

    print(f"writing to {OUTPUT_DIR}")
    for label, builder in builders:
        written = builder(args.formats)
        if written:
            print(f"  {label}: {', '.join(p.name for p in written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
