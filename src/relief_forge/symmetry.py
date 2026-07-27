"""Reconstruction of a mesh from one half of itself by planar reflection.

The operation clips the mesh at an axis-aligned plane, discards one side, and
reflects the retained half across that plane. Because reflection reverses
handedness, triangle winding is reversed at the same time, otherwise every
mirrored normal would point inward and the result would fail the signed-volume
test.

Applicability
-------------
This is appropriate **only when bilateral symmetry is a known property of the
object**. Where it holds, it is not an approximation: it replaces the
reconstruction of one half with the reconstruction of the other, which is
useful when one side was captured or reconstructed more cleanly. Where it does
not hold, it discards real asymmetric geometry and fabricates a false mirror.
The caller is responsible for that judgement; nothing here can verify it.

Reflecting front-to-back is a distinct operation with a much weaker
justification, since a surface reconstructed from a single viewpoint carries no
information about the reverse side.

Robustness
----------
Vertices lying exactly on the clipping plane are common on symmetric parts and
produce three degenerate cases, each handled explicitly:

* a triangle lying wholly in the plane would be duplicated by its own mirror
  image, creating a doubled interior wall, so it is discarded;
* a triangle on the discarded side touching the plane at a single vertex clips
  to a zero-area sliver, so it is removed by an area threshold;
* an intersection computed at a vertex already on the plane repeats that point
  in the output polygon, so consecutive duplicates are collapsed.

The clipped boundary lies exactly on the plane and is deliberately left open:
the reflected copy has an identical boundary, so welding the two halves closes
the seam without cap geometry.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .exceptions import InvalidPlaneError
from .stl_io import TriangleArray, bounding_diagonal

__all__ = ["AXIS_INDEX", "Axis", "clip_to_halfspace", "reflect", "symmetrize"]

#: Axis names accepted by the public functions.
Axis = Literal["x", "y", "z"]

#: Mapping from axis name to coordinate column.
AXIS_INDEX: dict[str, int] = {"x": 0, "y": 1, "z": 2}

_PLANE_TOLERANCE = 1e-7
_AREA_TOLERANCE = 1e-6


def _resolve_axis(axis: Axis | int) -> int:
    if isinstance(axis, int):
        if axis not in (0, 1, 2):
            raise InvalidPlaneError(f"axis index must be 0, 1 or 2, got {axis}")
        return axis
    try:
        return AXIS_INDEX[axis]
    except KeyError as exc:
        raise InvalidPlaneError(
            f"axis must be one of {sorted(AXIS_INDEX)}, got {axis!r}"
        ) from exc


def plane_position(triangles: TriangleArray, axis: Axis | int) -> float:
    """Return the bounding-box midpoint along `axis`, the default cut plane."""
    index = _resolve_axis(axis)
    values = triangles.reshape(-1, 3)[:, index]
    return float((values.min() + values.max()) / 2.0)


def _append_unique(
    polygon: list[np.ndarray], point: np.ndarray, epsilon: float
) -> None:
    """Append `point` unless it duplicates the previous entry.

    An intersection computed at a vertex already lying on the plane reproduces
    that vertex, which would otherwise create a zero-length polygon edge.
    """
    if polygon and np.linalg.norm(point - polygon[-1]) < epsilon:
        return
    polygon.append(point)


def clip_to_halfspace(
    triangles: TriangleArray,
    axis: Axis | int,
    position: float,
    *,
    keep_positive: bool = False,
) -> TriangleArray:
    """Clip triangles against an axis-aligned plane, keeping one side.

    Implements Sutherland-Hodgman clipping per triangle. Triangles crossing the
    plane yield a quadrilateral or triangle, which is fan-triangulated. Winding
    is preserved, so a consistently wound input remains consistently wound.

    Parameters
    ----------
    triangles : ndarray, shape (n_triangles, 3, 3)
        Triangle corners.
    axis : {'x', 'y', 'z'} or int
        Plane normal direction.
    position : float
        Plane offset along `axis`.
    keep_positive : bool, optional
        Retain the half with coordinates greater than `position`.

    Returns
    -------
    ndarray, shape (m_triangles, 3, 3)
        The retained portion, with degenerate output removed.
    """
    index = _resolve_axis(axis)
    diagonal = bounding_diagonal(triangles)
    plane_epsilon = diagonal * _PLANE_TOLERANCE
    area_epsilon = (diagonal * _AREA_TOLERANCE) ** 2
    orientation = 1.0 if keep_positive else -1.0

    kept: list[np.ndarray] = []

    for triangle in triangles:
        distance = orientation * (triangle[:, index] - position)
        distance = np.where(np.abs(distance) < plane_epsilon, 0.0, distance)

        if np.all(distance == 0.0):
            continue  # coplanar with the cut; its mirror would duplicate it

        inside = distance >= 0.0
        inside_count = int(inside.sum())
        if inside_count == 0:
            continue
        if inside_count == 3:
            kept.append(triangle)
            continue

        polygon: list[np.ndarray] = []

        for i in range(3):
            j = (i + 1) % 3
            if inside[i]:
                _append_unique(polygon, triangle[i], plane_epsilon)
            # A vertex exactly on the plane is emitted above; only a true sign
            # change produces an additional intersection point.
            crosses = (distance[i] > 0) != (distance[j] > 0)
            if crosses and distance[i] != 0.0 and distance[j] != 0.0:
                t = distance[i] / (distance[i] - distance[j])
                point = triangle[i] + t * (triangle[j] - triangle[i])
                point[index] = position  # snap exactly onto the plane
                _append_unique(polygon, point, plane_epsilon)

        if len(polygon) > 1 and np.linalg.norm(polygon[0] - polygon[-1]) < plane_epsilon:
            polygon.pop()
        if len(polygon) < 3:
            continue

        fan = np.asarray(polygon)
        for k in range(1, len(fan) - 1):
            kept.append(np.stack([fan[0], fan[k], fan[k + 1]]))

    if not kept:
        return np.zeros((0, 3, 3), dtype=np.float64)

    result = np.asarray(kept, dtype=np.float64)
    a, b, c = result[:, 0], result[:, 1], result[:, 2]
    areas = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2.0
    return result[areas > area_epsilon]


def reflect(triangles: TriangleArray, axis: Axis | int, position: float) -> TriangleArray:
    """Mirror triangles across an axis-aligned plane and reverse their winding.

    Reflection inverts handedness, so winding must be reversed for normals to
    continue pointing outward.
    """
    index = _resolve_axis(axis)
    mirrored = triangles.copy()
    mirrored[:, :, index] = 2.0 * position - mirrored[:, :, index]
    return mirrored[:, ::-1, :]


def symmetrize(
    triangles: TriangleArray,
    axis: Axis | int = "x",
    position: float | None = None,
    *,
    keep_positive: bool = False,
) -> TriangleArray:
    """Replace a mesh with one half of itself and that half's mirror image.

    Parameters
    ----------
    triangles : ndarray, shape (n_triangles, 3, 3)
        Triangle corners.
    axis : {'x', 'y', 'z'} or int, optional
        Mirror-plane normal. Defaults to ``'x'`` (left/right).
    position : float, optional
        Plane offset. Defaults to the bounding-box midpoint along `axis`.
    keep_positive : bool, optional
        Retain the half with coordinates greater than `position`.

    Returns
    -------
    ndarray, shape (m_triangles, 3, 3)
        Symmetric mesh. Closed and consistently wound whenever the input was.

    Raises
    ------
    InvalidPlaneError
        If the plane leaves no geometry on the retained side.
    """
    index = _resolve_axis(axis)
    if position is None:
        position = plane_position(triangles, index)

    half = clip_to_halfspace(triangles, index, position, keep_positive=keep_positive)
    if len(half) == 0:
        name = "xyz"[index]
        raise InvalidPlaneError(
            f"no geometry retained when cutting at {name}={position:g}; "
            "the mesh may lie entirely on the discarded side"
        )
    return np.concatenate([half, reflect(half, index, position)])
