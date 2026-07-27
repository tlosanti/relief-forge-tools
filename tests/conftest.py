"""Deterministic synthetic geometry with known analytical properties.

Every fixture is constructed in closed form rather than loaded from a file, so
the suite runs without network access, GPU, model weights or binary assets, and
expected values are derived from geometry rather than from a previous run of
the code under test.

Naming follows the defect each fixture exercises. ``closed_cube`` is the
reference solid; the remaining cube variants introduce exactly one defect each.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

# --------------------------------------------------------------------------
# Analytical constants shared between fixtures and assertions
# --------------------------------------------------------------------------

CUBE_EDGE_MM = 10.0
CUBE_VOLUME_MM3 = CUBE_EDGE_MM**3
CUBE_AREA_MM2 = 6.0 * CUBE_EDGE_MM**2

#: A cube tessellated as two triangles per face has 18 unique edges: the 12
#: geometric edges meet at 90 degrees and the 6 face diagonals at 0 degrees.
CUBE_MEAN_DIHEDRAL_DEG = (12 * 90.0 + 6 * 0.0) / 18.0

ANNULUS_OUTER_MM = 10.0
ANNULUS_INNER_MM = 4.0
ANNULUS_HEIGHT_MM = 3.0
ANNULUS_SEGMENTS = 64
ANNULUS_EXACT_VOLUME_MM3 = (
    math.pi * (ANNULUS_OUTER_MM**2 - ANNULUS_INNER_MM**2) * ANNULUS_HEIGHT_MM
)

#: Relative tolerance for values that are exact in floating point.
EXACT_RTOL = 1e-9

#: Relative tolerance for a 64-segment polygonal approximation of a circle.
#: The inscribed polygon under-estimates area by 1 - sinc(1/n) ~ 0.08 percent.
POLYGON_RTOL = 3e-3


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------

def build_box(
    low: tuple[float, float, float] = (0.0, 0.0, 0.0),
    high: tuple[float, float, float] = (CUBE_EDGE_MM,) * 3,
) -> NDArray[np.float64]:
    """Return a closed, outward-wound axis-aligned box as 12 triangles."""
    low_a, high_a = np.asarray(low, float), np.asarray(high, float)
    corner = {
        bits: np.array([low_a[i] if bits[i] == 0 else high_a[i] for i in range(3)])
        for bits in itertools.product([0, 1], repeat=3)
    }
    quads = [
        ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)),  # -x
        ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)),  # +x
        ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)),  # -y
        ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)),  # +y
        ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)),  # -z
        ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)),  # +z
    ]
    triangles = []
    for a, b, c, d in quads:
        triangles.append((corner[a], corner[b], corner[c]))
        triangles.append((corner[a], corner[c], corner[d]))
    return np.asarray(triangles, dtype=np.float64)


def reverse_winding(triangles: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the same triangles with reversed vertex order."""
    return triangles[:, ::-1, :].copy()


def build_annulus(
    outer: float = ANNULUS_OUTER_MM,
    inner: float = ANNULUS_INNER_MM,
    height: float = ANNULUS_HEIGHT_MM,
    segments: int = ANNULUS_SEGMENTS,
) -> NDArray[np.float64]:
    """Return a closed genus-one solid: a washer with a through hole.

    Exercises the case where a valid mesh is not simply connected, so a
    topology checker cannot assume Euler characteristic 2.
    """
    triangles: list[tuple[NDArray, NDArray, NDArray]] = []
    angles = [2.0 * math.pi * i / segments for i in range(segments)]
    outer_ring = [(outer * math.cos(a), outer * math.sin(a)) for a in angles]
    inner_ring = [(inner * math.cos(a), inner * math.sin(a)) for a in angles]

    for i in range(segments):
        j = (i + 1) % segments
        o0 = np.array([*outer_ring[i], 0.0])
        o1 = np.array([*outer_ring[j], 0.0])
        o0t = np.array([*outer_ring[i], height])
        o1t = np.array([*outer_ring[j], height])
        i0 = np.array([*inner_ring[i], 0.0])
        i1 = np.array([*inner_ring[j], 0.0])
        i0t = np.array([*inner_ring[i], height])
        i1t = np.array([*inner_ring[j], height])

        triangles += [(o0, o1, o1t), (o0, o1t, o0t)]  # outer wall, facing out
        triangles += [(i0, i1t, i1), (i0, i0t, i1t)]  # inner wall, facing in
        triangles += [(o0, i0, i1), (o0, i1, o1)]  # bottom annular ring
        triangles += [(o0t, o1t, i1t), (o0t, i1t, i0t)]  # top annular ring

    return np.asarray(triangles, dtype=np.float64)


def build_sphere(
    radius: float = 8.0,
    subdivisions: int = 3,
    offset: tuple[float, float, float] = (2.0, 0.0, 0.0),
) -> NDArray[np.float64]:
    """Return a subdivided octahedron approximating a sphere, offset from origin.

    The offset makes the solid asymmetric about the origin, which is what the
    symmetry tests need.
    """
    vertices = [
        np.array(v, dtype=float)
        for v in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    ]
    faces = [
        (0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
        (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5),
    ]
    triangles = [(vertices[a], vertices[b], vertices[c]) for a, b, c in faces]

    for _ in range(subdivisions):
        refined = []
        for a, b, c in triangles:
            ab = (a + b) / np.linalg.norm(a + b)
            bc = (b + c) / np.linalg.norm(b + c)
            ca = (c + a) / np.linalg.norm(c + a)
            refined += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
        triangles = refined

    shift = np.asarray(offset, dtype=float)
    return np.asarray(
        [tuple(corner * radius + shift for corner in tri) for tri in triangles],
        dtype=np.float64,
    )


def write_ascii_stl(path: Path, triangles: NDArray[np.float64]) -> Path:
    """Write triangles as ASCII STL, for parser round-trip testing."""
    lines = ["solid test"]
    for triangle in triangles:
        lines.append("facet normal 0 0 0")
        lines.append("  outer loop")
        for corner in triangle:
            lines.append(f"    vertex {corner[0]:.9f} {corner[1]:.9f} {corner[2]:.9f}")
        lines.append("  endloop")
        lines.append("endfacet")
    lines.append("endsolid test")
    path.write_text("\n".join(lines))
    return path


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def closed_cube() -> NDArray[np.float64]:
    """Reference solid: closed, manifold, outward-wound."""
    return build_box()


@pytest.fixture
def inverted_cube() -> NDArray[np.float64]:
    """Closed and manifold, but wound inward; signed volume is negative."""
    return reverse_winding(build_box())


@pytest.fixture
def cube_missing_face() -> NDArray[np.float64]:
    """Open surface: one face removed, leaving four boundary edges."""
    return build_box()[:-2]


@pytest.fixture
def cube_reversed_triangle() -> NDArray[np.float64]:
    """Closed surface with one triangle wound against its neighbours."""
    triangles = build_box()
    triangles[-1] = triangles[-1][::-1]
    return triangles


@pytest.fixture
def disconnected_solids() -> NDArray[np.float64]:
    """Two separated closed cubes: valid topology, two components."""
    return np.concatenate([build_box(), build_box(low=(50, 0, 0), high=(60, 10, 10))])


@pytest.fixture
def nonmanifold_flap() -> NDArray[np.float64]:
    """A closed cube with two extra triangles sharing one of its edges."""
    cube = build_box()
    flap = build_box(low=(0, 0, 10), high=(10, 10, 20))[:2]
    return np.concatenate([cube, flap])


@pytest.fixture
def annulus() -> NDArray[np.float64]:
    """Closed genus-one solid with a through hole."""
    return build_annulus()


@pytest.fixture
def offset_sphere() -> NDArray[np.float64]:
    """Closed solid positioned asymmetrically about the origin."""
    return build_sphere()


@pytest.fixture
def asymmetric_box() -> NDArray[np.float64]:
    """Box spanning x in [-10, 5], used for off-centre symmetry cuts."""
    return build_box(low=(-10, -6, 0), high=(5, 6, 4))
