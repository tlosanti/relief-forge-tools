"""STL parsing, writing and vertex welding."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import (  # type: ignore[import-not-found]
    CUBE_EDGE_MM,
    EXACT_RTOL,
    build_box,
    write_ascii_stl,
)
from relief_forge import read_stl, weld, write_stl
from relief_forge.exceptions import DegenerateMeshError, STLParseError


class TestRoundTrip:
    def test_binary_round_trip_is_lossless_to_float32(self, closed_cube, tmp_path):
        path = tmp_path / "cube.stl"
        write_stl(path, closed_cube)
        assert np.allclose(read_stl(path), closed_cube, rtol=EXACT_RTOL)

    def test_ascii_and_binary_parse_identically(self, closed_cube, tmp_path):
        binary = tmp_path / "cube.stl"
        write_stl(binary, closed_cube)
        ascii_path = write_ascii_stl(tmp_path / "cube_ascii.stl", closed_cube)
        assert np.allclose(read_stl(binary), read_stl(ascii_path), atol=1e-6)

    def test_triangle_count_preserved(self, annulus, tmp_path):
        path = tmp_path / "annulus.stl"
        write_stl(path, annulus)
        assert len(read_stl(path)) == len(annulus)


class TestNormalHandling:
    def test_written_normals_agree_with_winding(self, closed_cube, tmp_path):
        """Stored normals are recomputed, never trusted from the source."""
        path = tmp_path / "cube.stl"
        write_stl(path, closed_cube)

        raw = path.read_bytes()
        record = np.frombuffer(
            raw, dtype=np.dtype([("f", "<12f4"), ("attr", "<u2")]), count=len(closed_cube), offset=84
        )
        stored = record["f"].reshape(-1, 4, 3)[:, 0, :]

        a, b, c = closed_cube[:, 0], closed_cube[:, 1], closed_cube[:, 2]
        expected = np.cross(b - a, c - a)
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)
        assert np.allclose(stored, expected, atol=1e-6)


class TestParserErrors:
    def test_truncated_binary_file_rejected(self, tmp_path):
        path = tmp_path / "short.stl"
        path.write_bytes(b"\0" * 40)
        with pytest.raises(STLParseError, match="too short"):
            read_stl(path)

    def test_small_ascii_file_is_accepted(self, tmp_path):
        """A one-triangle ASCII solid is under 84 bytes and must still parse."""
        path = tmp_path / "tiny.stl"
        path.write_text(
            "solid s\nfacet normal 0 0 0\nouter loop\n"
            "vertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\n"
            "endloop\nendfacet\nendsolid s\n"
        )
        assert read_stl(path).shape == (1, 3, 3)

    def test_garbage_rejected(self, tmp_path):
        path = tmp_path / "junk.stl"
        path.write_bytes(b"not an stl file" * 20)
        with pytest.raises(STLParseError):
            read_stl(path)

    def test_ascii_with_partial_triangle_rejected(self, tmp_path):
        path = tmp_path / "partial.stl"
        path.write_text(
            "solid s\n" + "vertex 0 0 0\nvertex 1 0 0\n" * 1 + "x" * 100 + "\nendsolid s\n"
        )
        with pytest.raises(STLParseError, match="multiple of three"):
            read_stl(path)

    def test_binary_solid_prefix_not_misread_as_ascii(self, closed_cube, tmp_path):
        """Binary files often begin with the word 'solid'; length decides format."""
        path = tmp_path / "tricky.stl"
        write_stl(path, closed_cube)
        raw = bytearray(path.read_bytes())
        raw[0:5] = b"solid"
        path.write_bytes(bytes(raw))
        assert len(read_stl(path)) == len(closed_cube)


class TestWelding:
    def test_cube_welds_to_eight_vertices(self, closed_cube):
        """36 stored corners collapse to the cube's 8 distinct positions."""
        result = weld(closed_cube)
        assert len(result.vertices) == 8
        assert result.faces.shape == (12, 3)

    def test_faces_reconstruct_original_geometry(self, closed_cube):
        result = weld(closed_cube)
        assert np.allclose(result.vertices[result.faces], closed_cube, rtol=EXACT_RTOL)

    def test_tolerance_scales_with_model_size(self):
        small = weld(build_box(high=(1.0, 1.0, 1.0)))
        large = weld(build_box(high=(1000.0, 1000.0, 1000.0)))
        assert large.tolerance == pytest.approx(small.tolerance * 1000.0, rel=EXACT_RTOL)

    def test_empty_mesh_rejected(self):
        with pytest.raises(DegenerateMeshError, match="no triangles"):
            weld(np.zeros((0, 3, 3)))

    def test_zero_extent_mesh_rejected(self):
        degenerate = np.zeros((1, 3, 3))
        with pytest.raises(DegenerateMeshError, match="no extent"):
            weld(degenerate)

    def test_near_coincident_vertices_merge(self):
        """Vertices closer than the tolerance are treated as one."""
        cube = build_box()
        nudged = cube.copy()
        nudged[0, 0, 0] += CUBE_EDGE_MM * 1e-9
        assert len(weld(nudged).vertices) == len(weld(cube).vertices) == 8
