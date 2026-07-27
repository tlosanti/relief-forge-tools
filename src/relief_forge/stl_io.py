"""STL reading, writing and vertex welding.

STL files store three independent vertex positions per triangle and carry no
index buffer, so a file that describes a closed solid is indistinguishable from
one describing loose triangles until coincident vertices are merged. Every
topological query in :mod:`relief_forge.topology` therefore operates on the
output of :func:`weld`, never on raw triangle soup.

Face normals recorded in the file are ignored on read and recomputed on write,
because exporters frequently store normals that disagree with vertex winding.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .exceptions import DegenerateMeshError, STLParseError

__all__ = ["TriangleArray", "WeldResult", "read_stl", "weld", "write_stl"]

#: Triangle corner array of shape ``(n_triangles, 3, 3)``, dtype ``float64``.
TriangleArray = NDArray[np.float64]

_BINARY_HEADER_BYTES = 80
_BINARY_COUNT_BYTES = 4
_BINARY_RECORD_BYTES = 50
_RECORD_DTYPE = np.dtype([("f", "<12f4"), ("attr", "<u2")])


@dataclass(frozen=True)
class WeldResult:
    """Indexed mesh produced by merging coincident triangle corners.

    Attributes
    ----------
    vertices : ndarray, shape (n_vertices, 3)
        Unique vertex positions.
    faces : ndarray, shape (n_triangles, 3)
        Indices into `vertices`, preserving the original winding order.
    tolerance : float
        Absolute snapping distance used, in model units.
    """

    vertices: NDArray[np.float64]
    faces: NDArray[np.int64]
    tolerance: float


def read_stl(path: str | Path) -> TriangleArray:
    """Read a binary or ASCII STL file.

    Format is detected by checking whether the declared triangle count is
    consistent with the file length, which is more reliable than testing for a
    leading ``solid`` keyword: many binary writers emit that string too.

    Parameters
    ----------
    path : str or Path
        File to read.

    Returns
    -------
    ndarray, shape (n_triangles, 3, 3)
        Triangle corners in file order.

    Raises
    ------
    STLParseError
        If the file matches neither format.
    """
    path = Path(path)
    raw = path.read_bytes()

    minimum = _BINARY_HEADER_BYTES + _BINARY_COUNT_BYTES
    if len(raw) < minimum:
        # Too short for a binary header, but a small ASCII file is still valid:
        # a single-triangle ASCII solid is well under 84 bytes.
        try:
            return _read_ascii(raw.decode("utf-8", errors="replace"))
        except STLParseError as exc:
            raise STLParseError(
                f"{path.name}: {len(raw)} bytes is too short for a binary STL "
                f"header and ASCII parsing failed: {exc}"
            ) from exc

    declared = struct.unpack("<I", raw[_BINARY_HEADER_BYTES:minimum])[0]
    if len(raw) == minimum + declared * _BINARY_RECORD_BYTES:
        return _read_binary(raw, declared)

    try:
        return _read_ascii(raw.decode("utf-8", errors="replace"))
    except STLParseError as exc:
        raise STLParseError(
            f"{path.name}: not binary STL (header declares {declared} triangles, "
            f"implying {minimum + declared * _BINARY_RECORD_BYTES} bytes, file has "
            f"{len(raw)}) and ASCII parsing failed: {exc}"
        ) from exc


def _read_binary(raw: bytes, count: int) -> TriangleArray:
    """Decode binary STL records, discarding the stored normal of each facet."""
    records = np.frombuffer(
        raw, dtype=_RECORD_DTYPE, count=count, offset=_BINARY_HEADER_BYTES + _BINARY_COUNT_BYTES
    )
    corners = records["f"].reshape(count, 4, 3)[:, 1:, :]
    return np.ascontiguousarray(corners, dtype=np.float64)


def _read_ascii(text: str) -> TriangleArray:
    """Decode ASCII STL by collecting ``vertex`` records in document order."""
    corners: list[list[float]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("vertex"):
            parts = stripped.split()
            if len(parts) < 4:
                raise STLParseError(f"malformed vertex record: {stripped!r}")
            corners.append([float(value) for value in parts[1:4]])

    if not corners:
        raise STLParseError("no vertex records found")
    if len(corners) % 3 != 0:
        raise STLParseError(
            f"vertex count {len(corners)} is not a multiple of three"
        )
    return np.asarray(corners, dtype=np.float64).reshape(-1, 3, 3)


def write_stl(path: str | Path, triangles: TriangleArray) -> None:
    """Write triangles to a binary STL file.

    Facet normals are recomputed from vertex winding rather than carried
    through from any source, so the written normals always agree with the
    geometry. Degenerate triangles receive a zero normal.

    Parameters
    ----------
    path : str or Path
        Destination file.
    triangles : ndarray, shape (n_triangles, 3, 3)
        Triangle corners.
    """
    triangles = np.asarray(triangles, dtype=np.float64)
    count = len(triangles)

    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    normals = np.cross(b - a, c - a)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)

    records = np.zeros(count, dtype=_RECORD_DTYPE)
    records["f"] = (
        np.concatenate([normals[:, None, :], triangles], axis=1)
        .reshape(count, 12)
        .astype(np.float32)
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(b"\0" * _BINARY_HEADER_BYTES)
        handle.write(struct.pack("<I", count))
        handle.write(records.tobytes())


def bounding_diagonal(triangles: TriangleArray) -> float:
    """Return the length of the axis-aligned bounding-box diagonal."""
    flat = triangles.reshape(-1, 3)
    return float(np.linalg.norm(flat.max(axis=0) - flat.min(axis=0)))


def weld(triangles: TriangleArray, relative_tolerance: float = 1e-6) -> WeldResult:
    """Merge coincident triangle corners into a shared vertex list.

    Positions are snapped to a grid whose spacing is `relative_tolerance` times
    the bounding-box diagonal, so the effective tolerance scales with the part
    and behaves the same on a 3 mm setting as on a 300 mm sculpture.

    Parameters
    ----------
    triangles : ndarray, shape (n_triangles, 3, 3)
        Triangle corners.
    relative_tolerance : float, optional
        Snapping grid spacing as a fraction of the bounding-box diagonal.

    Returns
    -------
    WeldResult
        Indexed mesh and the absolute tolerance applied.

    Raises
    ------
    DegenerateMeshError
        If the mesh is empty or has zero spatial extent.
    """
    if triangles.size == 0:
        raise DegenerateMeshError("mesh contains no triangles")

    diagonal = bounding_diagonal(triangles)
    if diagonal <= 0.0:
        raise DegenerateMeshError("all vertices coincide; mesh has no extent")

    tolerance = diagonal * relative_tolerance
    flat = triangles.reshape(-1, 3)
    keys = np.round(flat / tolerance).astype(np.int64)

    _, first_index, inverse = np.unique(
        keys, axis=0, return_index=True, return_inverse=True
    )
    return WeldResult(
        vertices=flat[first_index],
        faces=inverse.reshape(-1, 3).astype(np.int64),
        tolerance=tolerance,
    )
