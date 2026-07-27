"""Edge-incidence and connectivity analysis of indexed triangle meshes.

A triangle mesh bounds a solid region if and only if its surface is closed and
consistently oriented. Both properties are decided here by directed-edge
bookkeeping: each triangle ``(v0, v1, v2)`` contributes the directed edges
``(v0, v1)``, ``(v1, v2)`` and ``(v2, v0)``, and on a closed, consistently
wound surface every directed edge occurs exactly once together with exactly one
occurrence of its reverse.

Deviations classify the defect rather than merely reporting failure:

boundary edge
    Used by one triangle only. The surface has a hole.
non-manifold edge
    Used by more than two triangles. The surface self-intersects or contains
    an internal partition.
inconsistently wound edge
    Used by two triangles that traverse it in the same direction, meaning the
    neighbours disagree about which side is outside.
degenerate edge
    Both endpoints weld to the same vertex, so the triangle has no area.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

__all__ = ["TopologyReport", "analyse_topology", "count_components", "edge_incidence"]


@dataclass(frozen=True)
class TopologyReport:
    """Result of edge-incidence and connectivity analysis.

    Attributes
    ----------
    unique_edges : int
        Number of distinct undirected edges.
    boundary_edges : int
        Edges incident to exactly one triangle.
    nonmanifold_edges : int
        Edges incident to more than two triangles.
    inconsistent_edges : int
        Two-triangle edges whose neighbours wind in the same direction.
    degenerate_edges : int
        Edges whose endpoints are the same vertex.
    components : int
        Number of connected components over triangle adjacency.
    """

    unique_edges: int
    boundary_edges: int
    nonmanifold_edges: int
    inconsistent_edges: int
    degenerate_edges: int
    components: int

    #: Field order used when rendering defect summaries. Class-level, so it is
    #: not a constructor argument.
    _DEFECTS: ClassVar[tuple[str, ...]] = (
        "boundary_edges",
        "nonmanifold_edges",
        "inconsistent_edges",
        "degenerate_edges",
    )

    @property
    def is_closed_manifold(self) -> bool:
        """True when the surface is closed, manifold and consistently wound."""
        return (
            self.boundary_edges == 0
            and self.nonmanifold_edges == 0
            and self.inconsistent_edges == 0
            and self.degenerate_edges == 0
        )

    def defects(self) -> dict[str, int]:
        """Return the non-zero defect counts, keyed by field name."""
        return {name: getattr(self, name) for name in self._DEFECTS if getattr(self, name)}

    def to_dict(self) -> dict[str, int | bool]:
        """Return a JSON-serialisable representation."""
        return {
            "unique_edges": self.unique_edges,
            "boundary_edges": self.boundary_edges,
            "nonmanifold_edges": self.nonmanifold_edges,
            "inconsistent_edges": self.inconsistent_edges,
            "degenerate_edges": self.degenerate_edges,
            "components": self.components,
            "is_closed_manifold": self.is_closed_manifold,
        }


def edge_incidence(faces: NDArray[np.int64]) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Return unique undirected edges with forward and reverse use counts.

    Parameters
    ----------
    faces : ndarray, shape (n_triangles, 3)
        Vertex indices per triangle.

    Returns
    -------
    unique : ndarray, shape (n_edges, 2)
        Undirected edges as sorted index pairs.
    forward : ndarray, shape (n_edges,)
        Times each edge was traversed low index to high index.
    reverse : ndarray, shape (n_edges,)
        Times each edge was traversed high index to low index.
    """
    directed = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]
    )
    low = np.minimum(directed[:, 0], directed[:, 1])
    high = np.maximum(directed[:, 0], directed[:, 1])
    traversed_forward = directed[:, 0] < directed[:, 1]

    undirected = np.stack([low, high], axis=1)
    unique, inverse = np.unique(undirected, axis=0, return_inverse=True)

    forward = np.bincount(
        inverse, weights=traversed_forward.astype(np.int64), minlength=len(unique)
    ).astype(np.int64)
    total = np.bincount(inverse, minlength=len(unique)).astype(np.int64)
    return unique, forward, total - forward


def count_components(faces: NDArray[np.int64], n_vertices: int) -> int:
    """Count connected components using union-find over triangle corners.

    A single solid yields one component. Higher counts indicate detached
    fragments, which for generated meshes usually means floating debris, and
    for boolean results usually means the union did not fuse.

    Parameters
    ----------
    faces : ndarray, shape (n_triangles, 3)
        Vertex indices per triangle.
    n_vertices : int
        Total vertex count.

    Returns
    -------
    int
        Number of components containing at least one triangle.
    """
    parent = np.arange(n_vertices, dtype=np.int64)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]  # path halving
            node = int(parent[node])
        return node

    for triangle in faces:
        first = find(int(triangle[0]))
        for corner in triangle[1:]:
            root = find(int(corner))
            if root != first:
                parent[root] = first

    referenced = np.unique(faces)
    return len({find(int(vertex)) for vertex in referenced})


def analyse_topology(faces: NDArray[np.int64], n_vertices: int) -> TopologyReport:
    """Classify every edge and count connected components.

    Parameters
    ----------
    faces : ndarray, shape (n_triangles, 3)
        Vertex indices per triangle, from :func:`relief_forge.stl_io.weld`.
    n_vertices : int
        Total vertex count.

    Returns
    -------
    TopologyReport
        Edge classification and component count.
    """
    unique, forward, reverse = edge_incidence(faces)
    total = forward + reverse

    return TopologyReport(
        unique_edges=len(unique),
        boundary_edges=int(np.count_nonzero(total == 1)),
        nonmanifold_edges=int(np.count_nonzero(total > 2)),
        inconsistent_edges=int(np.count_nonzero((total == 2) & (forward != reverse))),
        degenerate_edges=int(np.count_nonzero(unique[:, 0] == unique[:, 1])),
        components=count_components(faces, n_vertices),
    )
