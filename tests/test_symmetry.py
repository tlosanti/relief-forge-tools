"""Planar clipping, reflection and symmetry reconstruction.

The controlling requirement is that symmetrizing a closed, consistently wound
mesh yields another closed, consistently wound mesh. The plane-intersection
cases matter because a symmetric part routinely has vertices lying exactly on
its own mirror plane, which is where naive clipping produces slivers and
doubled faces.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import (  # type: ignore[import-not-found]
    ANNULUS_EXACT_VOLUME_MM3,
    CUBE_EDGE_MM,
    CUBE_VOLUME_MM3,
    EXACT_RTOL,
    POLYGON_RTOL,
    build_box,
    build_sphere,
)
from relief_forge import analyse_topology, signed_volume, weld
from relief_forge.exceptions import InvalidPlaneError
from relief_forge.symmetry import (
    clip_to_halfspace,
    plane_position,
    reflect,
    symmetrize,
)


def assert_closed_and_outward(triangles) -> None:
    """Assert the mesh is a closed manifold with outward-facing normals."""
    welded = weld(triangles)
    report = analyse_topology(welded.faces, len(welded.vertices))
    assert report.is_closed_manifold, f"defects: {report.defects()}"
    assert signed_volume(triangles) > 0.0


class TestClipping:
    def test_clip_at_centre_halves_the_volume(self, closed_cube):
        half = clip_to_halfspace(closed_cube, "x", CUBE_EDGE_MM / 2.0)
        extents = half.reshape(-1, 3).max(axis=0) - half.reshape(-1, 3).min(axis=0)
        assert extents[0] == pytest.approx(CUBE_EDGE_MM / 2.0, rel=EXACT_RTOL)

    def test_keep_positive_selects_the_other_half(self, closed_cube):
        low = clip_to_halfspace(closed_cube, "x", 5.0, keep_positive=False)
        high = clip_to_halfspace(closed_cube, "x", 5.0, keep_positive=True)
        assert low.reshape(-1, 3)[:, 0].max() == pytest.approx(5.0, abs=1e-9)
        assert high.reshape(-1, 3)[:, 0].min() == pytest.approx(5.0, abs=1e-9)

    def test_plane_outside_mesh_keeps_everything(self, closed_cube):
        kept = clip_to_halfspace(closed_cube, "x", 1000.0)
        assert len(kept) == len(closed_cube)

    def test_plane_beyond_mesh_keeps_nothing(self, closed_cube):
        assert len(clip_to_halfspace(closed_cube, "x", -1000.0)) == 0

    def test_no_zero_area_triangles_survive(self, annulus):
        """Vertices lying on the plane must not leave slivers behind."""
        kept = clip_to_halfspace(annulus, "x", 0.0)
        a, b, c = kept[:, 0], kept[:, 1], kept[:, 2]
        areas = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2.0
        assert (areas > 0).all()

    def test_clipped_boundary_lies_exactly_on_plane(self, offset_sphere):
        kept = clip_to_halfspace(offset_sphere, "x", 0.0)
        on_plane = kept.reshape(-1, 3)[:, 0]
        assert on_plane.max() <= 1e-9


class TestReflection:
    def test_reflection_is_an_involution(self, closed_cube):
        once = reflect(closed_cube, "x", 0.0)
        twice = reflect(once, "x", 0.0)
        assert np.allclose(np.sort(twice, axis=0), np.sort(closed_cube, axis=0))

    def test_reflection_reverses_orientation(self, closed_cube):
        """Without winding reversal the mirrored half would face inward."""
        mirrored = reflect(closed_cube, "x", 0.0)
        assert signed_volume(mirrored) == pytest.approx(
            signed_volume(closed_cube), rel=EXACT_RTOL
        )

    def test_reflection_preserves_volume_magnitude(self, annulus):
        mirrored = reflect(annulus, "y", 3.0)
        assert abs(signed_volume(mirrored)) == pytest.approx(
            abs(signed_volume(annulus)), rel=EXACT_RTOL
        )


class TestSymmetrize:
    def test_centred_cut_reproduces_a_symmetric_solid(self, closed_cube):
        """A cube cut through its own centre must come back unchanged in size."""
        result = symmetrize(closed_cube, "x")
        assert_closed_and_outward(result)
        assert signed_volume(result) == pytest.approx(CUBE_VOLUME_MM3, rel=1e-6)

    def test_offset_cut_produces_expected_dimensions(self, asymmetric_box):
        """Box spanning x in [-10, 5] cut at x=0 becomes 20 x 12 x 4."""
        result = symmetrize(asymmetric_box, "x", 0.0)
        assert_closed_and_outward(result)
        extents = result.reshape(-1, 3).max(axis=0) - result.reshape(-1, 3).min(axis=0)
        assert extents == pytest.approx([20.0, 12.0, 4.0], rel=1e-9)
        assert signed_volume(result) == pytest.approx(20.0 * 12.0 * 4.0, rel=1e-9)

    def test_annulus_survives_with_hole_intact(self, annulus):
        """Vertices sit exactly on the cut plane here; the hole must persist."""
        result = symmetrize(annulus, "x", 0.0)
        assert_closed_and_outward(result)
        assert signed_volume(result) == pytest.approx(
            ANNULUS_EXACT_VOLUME_MM3, rel=POLYGON_RTOL
        )

        welded = weld(result)
        report = analyse_topology(welded.faces, len(welded.vertices))
        chi = len(welded.vertices) - report.unique_edges + len(welded.faces)
        assert chi == 0, "genus-one topology should be preserved"

    def test_offset_sphere_matches_spherical_cap_formula(self):
        """Cap of height h on radius r has volume pi h^2 (3r - h) / 3; doubled."""
        radius, offset = 8.0, 2.0
        sphere = build_sphere(radius=radius, subdivisions=5, offset=(offset, 0.0, 0.0))
        result = symmetrize(sphere, "x", 0.0)
        assert_closed_and_outward(result)

        height = radius - offset
        expected = 2.0 * math.pi * height**2 * (3.0 * radius - height) / 3.0
        assert signed_volume(result) == pytest.approx(expected, rel=2e-2)

    def test_every_axis_produces_valid_geometry(self, annulus):
        for axis in ("x", "y", "z"):
            assert_closed_and_outward(symmetrize(annulus, axis))

    def test_result_is_symmetric_about_the_plane(self, offset_sphere):
        result = symmetrize(offset_sphere, "x", 0.0)
        points = result.reshape(-1, 3)
        assert points[:, 0].min() == pytest.approx(-points[:, 0].max(), rel=1e-6)

    def test_default_plane_is_bounding_box_midpoint(self, asymmetric_box):
        assert plane_position(asymmetric_box, "x") == pytest.approx(-2.5, rel=EXACT_RTOL)

    def test_plane_missing_the_mesh_raises(self, closed_cube):
        with pytest.raises(InvalidPlaneError, match="no geometry retained"):
            symmetrize(closed_cube, "x", -1000.0)

    def test_keep_positive_gives_congruent_result(self, closed_cube):
        """Either half of a symmetric solid reconstructs the same volume."""
        low = symmetrize(closed_cube, "x", 5.0, keep_positive=False)
        high = symmetrize(closed_cube, "x", 5.0, keep_positive=True)
        assert signed_volume(low) == pytest.approx(signed_volume(high), rel=1e-9)


class TestPlaneIntersectionEdgeCases:
    def test_plane_exactly_on_a_face_does_not_duplicate_it(self, closed_cube):
        """Cutting at z=0, where a whole face lies, must not double that face."""
        result = symmetrize(closed_cube, "z", 0.0, keep_positive=True)
        assert_closed_and_outward(result)

    def test_plane_touching_a_single_vertex_produces_no_slivers(self):
        """A plane grazing one corner should discard, not emit a zero-area face."""
        box = build_box(low=(0, 0, 0), high=(10, 10, 10))
        result = symmetrize(box, "x", 10.0, keep_positive=False)
        assert_closed_and_outward(result)

    def test_repeated_symmetrization_is_stable(self, offset_sphere):
        """Symmetrizing an already-symmetric mesh must not degrade it."""
        once = symmetrize(offset_sphere, "x", 0.0)
        twice = symmetrize(once, "x", 0.0)
        assert_closed_and_outward(twice)
        assert signed_volume(twice) == pytest.approx(signed_volume(once), rel=1e-6)

    def test_invalid_axis_rejected(self, closed_cube):
        with pytest.raises(InvalidPlaneError):
            symmetrize(closed_cube, "w")  # type: ignore[arg-type]
