"""Geometric and physical measurement of triangle meshes.

Volume follows from the divergence theorem. For a closed surface of triangles
with corners :math:`\\mathbf{a}_i, \\mathbf{b}_i, \\mathbf{c}_i`,

.. math::

    V = \\frac{1}{6} \\sum_{i=1}^{N} \\mathbf{a}_i \\cdot
        (\\mathbf{b}_i \\times \\mathbf{c}_i)

The sign of :math:`V` is positive when triangles are wound counter-clockwise
seen from outside, so the same quantity tests orientation and measures volume.
Triangle area is

.. math::

    A_i = \\tfrac{1}{2}
    \\left\\| (\\mathbf{b}_i - \\mathbf{a}_i) \\times
              (\\mathbf{c}_i - \\mathbf{a}_i) \\right\\|

and estimated material mass is :math:`m = \\rho |V|`.

All lengths are assumed to be millimetres, matching the convention of STL
consumers in additive manufacturing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .stl_io import TriangleArray, WeldResult, read_stl, weld
from .topology import TopologyReport, analyse_topology

__all__ = [
    "DENSITIES_G_PER_CM3",
    "MeshReport",
    "analyse_mesh",
    "bounding_box",
    "material_masses",
    "mean_dihedral_angle",
    "signed_volume",
    "surface_area",
]

#: Nominal densities in g/cm^3. Alloy composition varies between suppliers, so
#: these are estimates for comparison rather than certified figures.
DENSITIES_G_PER_CM3: dict[str, float] = {
    "silver_925": 10.36,
    "bronze": 8.80,
    "gold_18k": 15.58,
    "resin": 1.15,
}

_MM3_PER_CM3 = 1000.0


def signed_volume(triangles: TriangleArray) -> float:
    """Return the signed volume enclosed by a triangle surface, in mm^3.

    Positive values indicate outward-facing winding. The result is meaningful
    only for closed surfaces; on an open surface it depends on the origin.
    """
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def surface_area(triangles: TriangleArray) -> float:
    """Return total surface area in mm^2."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    return float(np.linalg.norm(np.cross(b - a, c - a), axis=1).sum() / 2.0)


def bounding_box(triangles: TriangleArray) -> NDArray[np.float64]:
    """Return axis-aligned bounding-box extents ``[dx, dy, dz]`` in mm."""
    flat = triangles.reshape(-1, 3)
    return flat.max(axis=0) - flat.min(axis=0)


def mean_dihedral_angle(
    vertices: NDArray[np.float64], faces: NDArray[np.int64]
) -> float:
    """Return the mean angle between adjacent face normals, in degrees.

    This is a **limited detail-retention proxy**, not a quality metric. It is
    interpretable only when comparing closely related meshes of the same object
    at comparable triangle density and identical scale: a lower value then
    indicates that surface relief has been smoothed away. Across unrelated
    meshes, or across different tessellation densities, it carries no meaning.

    Parameters
    ----------
    vertices : ndarray, shape (n_vertices, 3)
        Vertex positions.
    faces : ndarray, shape (n_triangles, 3)
        Vertex indices per triangle.

    Returns
    -------
    float
        Mean dihedral angle over edges shared by exactly two triangles.
        Returns 0.0 when no such edge exists.
    """
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    normals = np.cross(b - a, c - a)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    usable = lengths[:, 0] > 0
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)

    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    owner = np.tile(np.arange(len(faces)), 3)
    _, inverse = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True)

    order = np.argsort(inverse, kind="stable")
    sorted_edges, sorted_owner = inverse[order], owner[order]

    # Adjacent entries sharing an edge identifier form a face pair.
    pair_start = np.flatnonzero(sorted_edges[:-1] == sorted_edges[1:])
    left, right = sorted_owner[pair_start], sorted_owner[pair_start + 1]
    keep = usable[left] & usable[right]
    if not np.any(keep):
        return 0.0

    cosine = np.clip(
        np.einsum("ij,ij->i", normals[left[keep]], normals[right[keep]]), -1.0, 1.0
    )
    return float(np.degrees(np.arccos(cosine)).mean())


def material_masses(volume_mm3: float) -> dict[str, float]:
    """Return estimated cast mass in grams for each material in the table.

    Assumes a fully dense solid and ignores sprue, flashing and shrinkage.
    """
    volume_cm3 = abs(volume_mm3) / _MM3_PER_CM3
    return {name: volume_cm3 * rho for name, rho in DENSITIES_G_PER_CM3.items()}


@dataclass(frozen=True)
class MeshReport:
    """Complete measurement and validation result for one mesh.

    Attributes
    ----------
    source : str
        Name of the analysed file.
    triangles : int
        Triangle count as stored.
    vertices : int
        Vertex count after welding.
    weld_tolerance_mm : float
        Absolute snapping distance used during welding.
    bbox_mm : tuple of float
        Axis-aligned extents.
    volume_mm3 : float
        Signed volume; sign encodes orientation.
    surface_area_mm2 : float
        Total triangle area.
    mean_dihedral_deg : float
        Detail-retention proxy; see :func:`mean_dihedral_angle`.
    topology : TopologyReport
        Edge classification and component count.
    expected_components : int
        Component count required to pass validation.
    """

    source: str
    triangles: int
    vertices: int
    weld_tolerance_mm: float
    bbox_mm: tuple[float, float, float]
    volume_mm3: float
    surface_area_mm2: float
    mean_dihedral_deg: float
    topology: TopologyReport
    expected_components: int = 1

    @property
    def is_outward_wound(self) -> bool:
        """True when total signed volume is positive."""
        return self.volume_mm3 > 0.0

    @property
    def has_expected_components(self) -> bool:
        """True when the component count matches the caller's expectation."""
        return self.topology.components == self.expected_components

    @property
    def passes_checks(self) -> bool:
        """True when the mesh satisfies every check this package performs.

        The conditions are: no boundary edges, no non-manifold edges, no
        degenerate edges, consistent local winding, positive signed volume, and
        the expected connected-component count.

        Passing indicates only that the mesh is a closed, coherently oriented
        solid. It does not establish minimum wall thickness, overhang
        supportability, printer tolerance, casting shrinkage, absence of
        trapped volumes, or any process-specific requirement.
        """
        return (
            self.topology.is_closed_manifold
            and self.is_outward_wound
            and self.has_expected_components
        )

    @property
    def volume_cm3(self) -> float:
        """Absolute volume in cm^3."""
        return abs(self.volume_mm3) / _MM3_PER_CM3

    def material_masses_g(self) -> dict[str, float]:
        """Estimated cast mass in grams per material."""
        return material_masses(self.volume_mm3)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "source": self.source,
            "triangles": self.triangles,
            "vertices": self.vertices,
            "weld_tolerance_mm": self.weld_tolerance_mm,
            "bbox_mm": list(self.bbox_mm),
            "volume_mm3": self.volume_mm3,
            "volume_cm3": self.volume_cm3,
            "surface_area_mm2": self.surface_area_mm2,
            "mean_dihedral_deg": self.mean_dihedral_deg,
            "outward_wound": self.is_outward_wound,
            "expected_components": self.expected_components,
            "has_expected_components": self.has_expected_components,
            "topology": self.topology.to_dict(),
            "material_mass_g": self.material_masses_g(),
            "passes_checks": self.passes_checks,
        }


def analyse_mesh(
    path: str | Path,
    *,
    expected_components: int = 1,
    relative_tolerance: float = 1e-6,
) -> MeshReport:
    """Read an STL file and produce a full measurement and validation report.

    Parameters
    ----------
    path : str or Path
        STL file to analyse.
    expected_components : int, optional
        Connected-component count required to pass validation.
    relative_tolerance : float, optional
        Vertex welding tolerance as a fraction of the bounding-box diagonal.

    Returns
    -------
    MeshReport
        Measurements, topology classification and pass/fail state.
    """
    path = Path(path)
    triangles = read_stl(path)
    welded: WeldResult = weld(triangles, relative_tolerance)
    extents = bounding_box(triangles)

    return MeshReport(
        source=path.name,
        triangles=len(triangles),
        vertices=len(welded.vertices),
        weld_tolerance_mm=welded.tolerance,
        bbox_mm=(float(extents[0]), float(extents[1]), float(extents[2])),
        volume_mm3=signed_volume(triangles),
        surface_area_mm2=surface_area(triangles),
        mean_dihedral_deg=mean_dihedral_angle(welded.vertices, welded.faces),
        topology=analyse_topology(welded.faces, len(welded.vertices)),
        expected_components=expected_components,
    )
