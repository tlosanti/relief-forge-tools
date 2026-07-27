"""Edge-incidence classification and connectivity.

Each defect fixture must be detected as exactly that defect, and the reference
solid must produce no false positives.
"""

from __future__ import annotations

from relief_forge import analyse_topology, weld
from relief_forge.topology import count_components, edge_incidence


def analyse(triangles):
    """Weld then classify, mirroring what the CLI does."""
    welded = weld(triangles)
    return analyse_topology(welded.faces, len(welded.vertices))


class TestValidSolids:
    def test_closed_cube_has_no_defects(self, closed_cube):
        report = analyse(closed_cube)
        assert report.is_closed_manifold
        assert report.defects() == {}
        assert report.components == 1

    def test_cube_edge_count_matches_euler(self, closed_cube):
        """V - E + F = 2 for a closed genus-zero surface: 8 - 18 + 12 = 2."""
        report = analyse(closed_cube)
        assert report.unique_edges == 18
        assert 8 - report.unique_edges + 12 == 2

    def test_inverted_cube_is_still_manifold(self, inverted_cube):
        """Reversing every triangle keeps the surface closed and consistent."""
        report = analyse(inverted_cube)
        assert report.is_closed_manifold

    def test_annulus_is_closed_despite_hole(self, annulus):
        """A through hole is valid geometry, not a topological defect."""
        report = analyse(annulus)
        assert report.is_closed_manifold
        assert report.components == 1

    def test_annulus_euler_characteristic_is_zero(self, annulus):
        """Genus one implies V - E + F = 0."""
        welded = weld(annulus)
        report = analyse_topology(welded.faces, len(welded.vertices))
        chi = len(welded.vertices) - report.unique_edges + len(welded.faces)
        assert chi == 0


class TestDefectDetection:
    def test_missing_face_produces_boundary_edges(self, cube_missing_face):
        report = analyse(cube_missing_face)
        assert not report.is_closed_manifold
        assert report.boundary_edges == 4
        assert report.nonmanifold_edges == 0
        assert report.inconsistent_edges == 0

    def test_reversed_triangle_detected_as_winding_defect(self, cube_reversed_triangle):
        """A flipped triangle leaves the surface closed but locally inconsistent."""
        report = analyse(cube_reversed_triangle)
        assert not report.is_closed_manifold
        assert report.inconsistent_edges == 3
        assert report.boundary_edges == 0

    def test_flap_produces_nonmanifold_edge(self, nonmanifold_flap):
        report = analyse(nonmanifold_flap)
        assert not report.is_closed_manifold
        assert report.nonmanifold_edges >= 1

    def test_defects_dictionary_lists_only_present_defects(self, cube_missing_face):
        defects = analyse(cube_missing_face).defects()
        assert set(defects) == {"boundary_edges"}


class TestConnectivity:
    def test_two_cubes_are_two_components(self, disconnected_solids):
        report = analyse(disconnected_solids)
        assert report.components == 2
        assert report.is_closed_manifold  # each solid is individually valid

    def test_single_cube_is_one_component(self, closed_cube):
        welded = weld(closed_cube)
        assert count_components(welded.faces, len(welded.vertices)) == 1


class TestEdgeIncidence:
    def test_every_edge_used_once_in_each_direction(self, closed_cube):
        welded = weld(closed_cube)
        _, forward, reverse = edge_incidence(welded.faces)
        assert (forward == 1).all()
        assert (reverse == 1).all()

    def test_directed_edge_total_is_three_per_triangle(self, annulus):
        welded = weld(annulus)
        _, forward, reverse = edge_incidence(welded.faces)
        assert int((forward + reverse).sum()) == 3 * len(welded.faces)
