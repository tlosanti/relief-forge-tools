"""Relief Forge Tools — mesh reconstruction and fabrication validation.

A geometry-processing toolkit that converts image-derived meshes into
validated, dimensionally scaled solids for small-scale casting and additive
manufacturing.

The package is organised so that the geometry core depends only on NumPy and is
independent of any mesh source:

``stl_io``
    Binary and ASCII STL parsing, writing and vertex welding.
``topology``
    Edge-incidence classification and connectivity analysis.
``measurements``
    Signed volume, surface area, bounding box and material mass estimation.
``symmetry``
    Planar clipping and reflection for bilaterally symmetric parts.
``generation``
    Optional, backend-agnostic image-to-mesh inference. Requires PyTorch.

Importing this package does not import PyTorch.
"""

from __future__ import annotations

from .exceptions import (
    BackendError,
    DegenerateMeshError,
    InvalidPlaneError,
    ReliefForgeError,
    STLParseError,
)
from .measurements import (
    DENSITIES_G_PER_CM3,
    MeshReport,
    analyse_mesh,
    bounding_box,
    material_masses,
    mean_dihedral_angle,
    signed_volume,
    surface_area,
)
from .stl_io import WeldResult, read_stl, weld, write_stl
from .symmetry import clip_to_halfspace, reflect, symmetrize
from .topology import TopologyReport, analyse_topology, count_components

__version__ = "0.2.0"

__all__ = [
    "__version__",
    # errors
    "ReliefForgeError",
    "STLParseError",
    "DegenerateMeshError",
    "InvalidPlaneError",
    "BackendError",
    # io
    "read_stl",
    "write_stl",
    "weld",
    "WeldResult",
    # topology
    "analyse_topology",
    "count_components",
    "TopologyReport",
    # measurement
    "analyse_mesh",
    "MeshReport",
    "signed_volume",
    "surface_area",
    "bounding_box",
    "mean_dihedral_angle",
    "material_masses",
    "DENSITIES_G_PER_CM3",
    # symmetry
    "symmetrize",
    "clip_to_halfspace",
    "reflect",
]
