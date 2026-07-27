"""Measurement accuracy against closed-form expected values.

Tolerances are stated explicitly at each assertion. Two regimes appear:

* ``EXACT_RTOL`` (1e-9) for polyhedra whose volume and area are exact in
  floating point, such as the axis-aligned cube;
* ``POLYGON_RTOL`` (3e-3) for the annulus, whose 64-segment outline inscribes
  the true circle and therefore under-estimates it by a known amount.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import (  # type: ignore[import-not-found]
    ANNULUS_EXACT_VOLUME_MM3,
    ANNULUS_HEIGHT_MM,
    ANNULUS_INNER_MM,
    ANNULUS_OUTER_MM,
    CUBE_AREA_MM2,
    CUBE_EDGE_MM,
    CUBE_MEAN_DIHEDRAL_DEG,
    CUBE_VOLUME_MM3,
    EXACT_RTOL,
    POLYGON_RTOL,
    build_box,
    build_sphere,
)
from relief_forge import analyse_mesh, weld, write_stl
from relief_forge.measurements import (
    DENSITIES_G_PER_CM3,
    bounding_box,
    material_masses,
    mean_dihedral_angle,
    signed_volume,
    surface_area,
)


class TestVolume:
    def test_cube_volume_is_exact(self, closed_cube):
        assert signed_volume(closed_cube) == pytest.approx(
            CUBE_VOLUME_MM3, rel=EXACT_RTOL
        )

    def test_inverted_cube_volume_is_negated(self, closed_cube, inverted_cube):
        assert signed_volume(inverted_cube) == pytest.approx(
            -signed_volume(closed_cube), rel=EXACT_RTOL
        )

    def test_volume_is_translation_invariant(self, closed_cube):
        """Volume must not depend on where the solid sits relative to the origin."""
        shifted = closed_cube + np.array([137.0, -42.0, 9.5])
        assert signed_volume(shifted) == pytest.approx(
            CUBE_VOLUME_MM3, rel=1e-9
        )

    def test_volume_scales_cubically(self):
        single = signed_volume(build_box(high=(1.0, 1.0, 1.0)))
        doubled = signed_volume(build_box(high=(2.0, 2.0, 2.0)))
        assert doubled == pytest.approx(8.0 * single, rel=EXACT_RTOL)

    def test_annulus_volume_matches_closed_form(self, annulus):
        """pi (R^2 - r^2) h, approached from below by the inscribed polygon."""
        computed = signed_volume(annulus)
        assert computed == pytest.approx(ANNULUS_EXACT_VOLUME_MM3, rel=POLYGON_RTOL)
        assert computed < ANNULUS_EXACT_VOLUME_MM3  # inscribed, so under-estimates

    def test_sphere_volume_matches_closed_form(self):
        radius = 8.0
        computed = signed_volume(build_sphere(radius=radius, subdivisions=5))
        exact = 4.0 / 3.0 * math.pi * radius**3
        assert computed == pytest.approx(exact, rel=1e-2)


class TestArea:
    def test_cube_area_is_exact(self, closed_cube):
        assert surface_area(closed_cube) == pytest.approx(CUBE_AREA_MM2, rel=EXACT_RTOL)

    def test_area_scales_quadratically(self):
        single = surface_area(build_box(high=(1.0, 1.0, 1.0)))
        tripled = surface_area(build_box(high=(3.0, 3.0, 3.0)))
        assert tripled == pytest.approx(9.0 * single, rel=EXACT_RTOL)

    def test_annulus_area_matches_closed_form(self, annulus):
        walls = 2.0 * math.pi * (ANNULUS_OUTER_MM + ANNULUS_INNER_MM) * ANNULUS_HEIGHT_MM
        rings = 2.0 * math.pi * (ANNULUS_OUTER_MM**2 - ANNULUS_INNER_MM**2)
        assert surface_area(annulus) == pytest.approx(walls + rings, rel=POLYGON_RTOL)


class TestBoundingBox:
    def test_cube_extents(self, closed_cube):
        assert bounding_box(closed_cube) == pytest.approx(
            [CUBE_EDGE_MM] * 3, rel=EXACT_RTOL
        )

    def test_annulus_extents(self, annulus):
        expected = [2 * ANNULUS_OUTER_MM, 2 * ANNULUS_OUTER_MM, ANNULUS_HEIGHT_MM]
        assert bounding_box(annulus) == pytest.approx(expected, rel=POLYGON_RTOL)


class TestDihedralAngle:
    def test_cube_mean_dihedral_matches_hand_count(self, closed_cube):
        """12 edges at 90 deg and 6 face diagonals at 0 deg average to 60 deg."""
        welded = weld(closed_cube)
        assert mean_dihedral_angle(welded.vertices, welded.faces) == pytest.approx(
            CUBE_MEAN_DIHEDRAL_DEG, rel=1e-9
        )

    def test_smoother_surface_scores_lower(self):
        """The proxy must move in the expected direction as a surface smooths."""
        coarse = build_sphere(subdivisions=2)
        fine = build_sphere(subdivisions=4)
        coarse_w, fine_w = weld(coarse), weld(fine)
        assert mean_dihedral_angle(fine_w.vertices, fine_w.faces) < mean_dihedral_angle(
            coarse_w.vertices, coarse_w.faces
        )

    def test_scale_invariant(self, closed_cube):
        welded_small = weld(closed_cube)
        welded_large = weld(closed_cube * 25.0)
        assert mean_dihedral_angle(
            welded_large.vertices, welded_large.faces
        ) == pytest.approx(
            mean_dihedral_angle(welded_small.vertices, welded_small.faces), rel=1e-9
        )


class TestMaterialMass:
    def test_masses_follow_density_times_volume(self):
        """m = rho |V|, with volume converted from mm^3 to cm^3."""
        masses = material_masses(CUBE_VOLUME_MM3)  # 1000 mm^3 = 1 cm^3
        for material, density in DENSITIES_G_PER_CM3.items():
            assert masses[material] == pytest.approx(density, rel=EXACT_RTOL)

    def test_mass_uses_absolute_volume(self):
        """An inward-wound mesh still has positive mass."""
        assert material_masses(-CUBE_VOLUME_MM3) == material_masses(CUBE_VOLUME_MM3)

    def test_gold_is_denser_than_silver_than_bronze_than_resin(self):
        masses = material_masses(CUBE_VOLUME_MM3)
        assert (
            masses["gold_18k"]
            > masses["silver_925"]
            > masses["bronze"]
            > masses["resin"]
        )


class TestMeshReport:
    def test_valid_cube_passes(self, closed_cube, tmp_path):
        path = tmp_path / "cube.stl"
        write_stl(path, closed_cube)
        report = analyse_mesh(path)
        assert report.passes_checks
        assert report.is_outward_wound
        assert report.volume_cm3 == pytest.approx(1.0, rel=1e-6)

    def test_inverted_cube_fails_on_orientation_only(self, inverted_cube, tmp_path):
        path = tmp_path / "inverted.stl"
        write_stl(path, inverted_cube)
        report = analyse_mesh(path)
        assert report.topology.is_closed_manifold
        assert not report.is_outward_wound
        assert not report.passes_checks

    def test_two_solids_fail_default_component_expectation(
        self, disconnected_solids, tmp_path
    ):
        path = tmp_path / "two.stl"
        write_stl(path, disconnected_solids)
        assert not analyse_mesh(path).passes_checks
        assert analyse_mesh(path, expected_components=2).passes_checks

    def test_report_serialises_to_json_types(self, closed_cube, tmp_path):
        import json

        path = tmp_path / "cube.stl"
        write_stl(path, closed_cube)
        payload = json.loads(json.dumps(analyse_mesh(path).to_dict()))
        assert payload["passes_checks"] is True
        assert payload["topology"]["boundary_edges"] == 0
