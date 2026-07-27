"""Exception hierarchy for relief_forge.

All library errors derive from :class:`ReliefForgeError`, so callers can
distinguish expected geometry or I/O failures from genuine bugs.
"""

from __future__ import annotations

__all__ = [
    "BackendError",
    "DegenerateMeshError",
    "InvalidPlaneError",
    "ReliefForgeError",
    "STLParseError",
]


class ReliefForgeError(Exception):
    """Base class for all errors raised by this package."""


class STLParseError(ReliefForgeError):
    """Raised when a file cannot be interpreted as binary or ASCII STL."""


class DegenerateMeshError(ReliefForgeError):
    """Raised when a mesh has no spatial extent or no usable triangles."""


class InvalidPlaneError(ReliefForgeError):
    """Raised when a clipping plane does not intersect the mesh usefully."""


class BackendError(ReliefForgeError):
    """Raised when a mesh-generation backend is unavailable or fails."""
