#!/usr/bin/env python3
"""
mesh_check.py — STL sanity + comparison harness for Relief Forge.

Pure numpy. No torch, no trimesh, no network. Runs anywhere python3 does,
including the shared .venv.

Follows the project rule of verifying geometry with a harness instead of
eyeballing it. The two assertions that matter, same as the existing STL check:

  1. Watertight   — every directed edge (u,v) has exactly one opposite (v,u).
  2. Outward wound — total signed volume is positive.

Usage
-----
  python3 tools/mesh_check.py piece.stl
  python3 tools/mesh_check.py forge.stl --compare hunyuan.stl
  python3 tools/mesh_check.py piece.stl --json

Exit code is 0 if the mesh is watertight and outward-wound, 1 otherwise, so
this can gate an export in a shell pipeline.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

# g/cm3. Sterling silver, typical casting bronze, 18k yellow gold, cast resin.
DENSITIES = {
    "silver_925": 10.36,
    "bronze": 8.80,
    "gold_18k": 15.58,
    "resin": 1.15,
}


# --------------------------------------------------------------------------
# STL reading
# --------------------------------------------------------------------------

def read_stl(path: Path) -> np.ndarray:
    """Return an (n_tri, 3, 3) float64 array of triangle corners.

    Detects binary vs ASCII by checking whether the declared triangle count
    matches the file length, which is more reliable than sniffing for the
    word 'solid' (plenty of binary STLs start with it).
    """
    raw = path.read_bytes()
    if len(raw) < 84:
        raise ValueError(f"{path.name}: too short to be an STL ({len(raw)} bytes)")

    n_declared = struct.unpack("<I", raw[80:84])[0]
    if len(raw) == 84 + n_declared * 50:
        return _read_binary(raw, n_declared)

    try:
        return _read_ascii(raw.decode("utf-8", errors="replace"))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"{path.name}: not a valid binary STL (expected "
            f"{84 + n_declared * 50} bytes, got {len(raw)}) and ASCII parse "
            f"failed: {exc}"
        ) from exc


def _read_binary(raw: bytes, n: int) -> np.ndarray:
    # Each record is 50 bytes: normal(3f) + 3 verts(3f each) + attr(uint16).
    # Read as 12 floats with a 2-byte tail, then drop the stored normal --
    # stored normals are frequently wrong, so we always recompute.
    rec = np.frombuffer(raw, dtype=np.dtype([("f", "<12f4"), ("attr", "<u2")]),
                        count=n, offset=84)
    return rec["f"].reshape(n, 4, 3)[:, 1:, :].astype(np.float64)


def write_stl(path: Path, tris: np.ndarray) -> None:
    """Write a binary STL. Normals are recomputed from winding, never stored
    blindly, since incoming normals are unreliable."""
    tris = np.asarray(tris, dtype=np.float64)
    n = len(tris)
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    nrm = np.cross(b - a, c - a)
    lens = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = np.divide(nrm, lens, out=np.zeros_like(nrm), where=lens > 0)

    rec = np.zeros(n, dtype=np.dtype([("f", "<12f4"), ("attr", "<u2")]))
    rec["f"] = np.concatenate(
        [nrm[:, None, :], tris], axis=1).reshape(n, 12).astype(np.float32)

    with open(path, "wb") as fh:
        fh.write(b"\0" * 80)
        fh.write(struct.pack("<I", n))
        fh.write(rec.tobytes())


def _read_ascii(text: str) -> np.ndarray:
    verts = [
        [float(p) for p in line.split()[1:4]]
        for line in text.splitlines()
        if line.strip().startswith("vertex")
    ]
    if not verts:
        raise ValueError("no 'vertex' lines found")
    if len(verts) % 3:
        raise ValueError(f"vertex count {len(verts)} is not a multiple of 3")
    return np.asarray(verts, dtype=np.float64).reshape(-1, 3, 3)


# --------------------------------------------------------------------------
# Topology
# --------------------------------------------------------------------------

def weld(tris: np.ndarray, rel_tol: float = 1e-6):
    """Merge coincident corners into shared indices.

    STL stores every triangle independently, so topology has to be rebuilt
    before any edge test means anything. Vertices are snapped to a grid sized
    relative to the bounding-box diagonal, which keeps the tolerance
    meaningful whether the piece is 3 mm or 300 mm.
    """
    flat = tris.reshape(-1, 3)
    diag = float(np.linalg.norm(flat.max(axis=0) - flat.min(axis=0)))
    if diag == 0:
        raise ValueError("degenerate mesh: all vertices coincide")

    scale = diag * rel_tol
    keys = np.round(flat / scale).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True,
                                  return_inverse=True)
    return flat[first], inverse.reshape(-1, 3), scale


def edge_report(faces: np.ndarray) -> dict:
    """Directed-edge twin test.

    A closed, consistently oriented surface has each directed edge exactly
    once, and its reverse exactly once. Anything else localises the defect:

      boundary    — an edge with no twin (a hole)
      nonmanifold — an edge used by more than two faces
      flipped     — a directed edge appearing twice (neighbours disagree
                    on winding, so the surface is closed but inside-out
                    somewhere)
    """
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])

    # Canonical undirected key, plus which direction this instance ran.
    lo = np.minimum(e[:, 0], e[:, 1])
    hi = np.maximum(e[:, 0], e[:, 1])
    forward = e[:, 0] < e[:, 1]

    und = np.stack([lo, hi], axis=1)
    uniq, inv = np.unique(und, axis=0, return_inverse=True)

    n_fwd = np.bincount(inv, weights=forward.astype(np.int64),
                        minlength=len(uniq))
    total = np.bincount(inv, minlength=len(uniq))
    n_rev = total - n_fwd

    boundary = int(np.sum(total == 1))
    nonmanifold = int(np.sum(total > 2))
    flipped = int(np.sum((total == 2) & (n_fwd != n_rev)))
    degenerate = int(np.sum(uniq[:, 0] == uniq[:, 1]))

    return {
        "edges": int(len(uniq)),
        "boundary_edges": boundary,
        "nonmanifold_edges": nonmanifold,
        "flipped_edges": flipped,
        "degenerate_edges": degenerate,
        "watertight": boundary == 0 and nonmanifold == 0 and flipped == 0
                      and degenerate == 0,
    }


def components(faces: np.ndarray, n_verts: int) -> int:
    """Count connected pieces via union-find over triangle corners.

    A relief with a bail should be 1. More than 1 usually means the union
    did not actually fuse, or the generator left floating debris.
    """
    parent = np.arange(n_verts)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, c in faces:
        for u, v in ((a, b), (b, c)):
            ru, rv = find(int(u)), find(int(v))
            if ru != rv:
                parent[ru] = rv

    used = np.unique(faces)
    return len({find(int(v)) for v in used})


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------

def signed_volume(tris: np.ndarray) -> float:
    """Divergence-theorem volume. Positive iff normals point outward."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def surface_area(tris: np.ndarray) -> float:
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    return float(np.linalg.norm(np.cross(b - a, c - a), axis=1).sum() / 2.0)


def roughness(verts: np.ndarray, faces: np.ndarray) -> float:
    """Mean dihedral angle (degrees) across shared edges.

    A proxy for how much surface ornament survived. A smooth blob sits near
    zero; filigree, beading and engraved lines push it up. Only meaningful
    when comparing two meshes of the *same* object at similar triangle
    density -- it is not an absolute quality score.
    """
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    normals = np.cross(b - a, c - a)
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    ok = lens[:, 0] > 0
    normals = np.divide(normals, lens, out=np.zeros_like(normals), where=lens > 0)

    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    owner = np.tile(np.arange(len(faces)), 3)
    und = np.sort(e, axis=1)
    _, inv = np.unique(und, axis=0, return_inverse=True)

    order = np.argsort(inv, kind="stable")
    inv_s, owner_s = inv[order], owner[order]
    # Adjacent entries sharing an edge id are a face pair.
    pair = np.flatnonzero(inv_s[:-1] == inv_s[1:])
    f0, f1 = owner_s[pair], owner_s[pair + 1]
    keep = ok[f0] & ok[f1]
    if not np.any(keep):
        return 0.0

    dot = np.clip(np.einsum("ij,ij->i", normals[f0[keep]], normals[f1[keep]]),
                  -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)).mean())


def analyse(path: Path) -> dict:
    tris = read_stl(path)
    verts, faces, tol = weld(tris)

    vol = signed_volume(tris)
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    edges = edge_report(faces)
    vol_cm3 = abs(vol) / 1000.0

    return {
        "file": path.name,
        "triangles": int(len(tris)),
        "vertices_welded": int(len(verts)),
        "weld_tolerance_mm": tol,
        "bbox_mm": [round(float(x), 3) for x in (hi - lo)],
        "volume_mm3": round(vol, 3),
        "outward_wound": vol > 0,
        "surface_area_mm2": round(surface_area(tris), 3),
        "roughness_deg": round(roughness(verts, faces), 3),
        "components": components(faces, len(verts)),
        **edges,
        "cast_weight_g": {k: round(vol_cm3 * d, 2) for k, d in DENSITIES.items()},
        "printable": edges["watertight"] and vol > 0,
    }


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def show(r: dict) -> None:
    mark = "PASS" if r["printable"] else "FAIL"
    print(f"\n=== {r['file']} — {mark} ===")
    print(f"  triangles      {r['triangles']:,}  ({r['vertices_welded']:,} welded verts)")
    bb = r["bbox_mm"]
    print(f"  bounding box   {bb[0]} x {bb[1]} x {bb[2]} mm")
    print(f"  volume         {r['volume_mm3'] / 1000.0:.3f} cm3")
    print(f"  surface area   {r['surface_area_mm2'] / 100.0:.2f} cm2")
    print(f"  roughness      {r['roughness_deg']:.2f} deg mean dihedral")
    print(f"  components     {r['components']}")

    print("  watertight     " + ("yes" if r["watertight"] else "NO"))
    if not r["watertight"]:
        for key, label in (
            ("boundary_edges", "holes (edges with no twin)"),
            ("nonmanifold_edges", "non-manifold edges (>2 faces)"),
            ("flipped_edges", "inconsistently wound edges"),
            ("degenerate_edges", "zero-length edges"),
        ):
            if r[key]:
                print(f"      {r[key]:,} {label}")
    print("  outward wound  " + ("yes" if r["outward_wound"] else "NO — normals inverted"))

    w = r["cast_weight_g"]
    print(f"  cast weight    silver {w['silver_925']} g | bronze {w['bronze']} g "
          f"| 18k gold {w['gold_18k']} g | resin {w['resin']} g")


def compare(a: dict, b: dict) -> None:
    print(f"\n=== {a['file']} vs {b['file']} ===")

    def row(label: str, x, y, fmt="{:.3f}") -> None:
        sx = fmt.format(x) if isinstance(x, float) else str(x)
        sy = fmt.format(y) if isinstance(y, float) else str(y)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and x:
            delta = f"{(y - x) / abs(x) * 100:+.1f}%"
        else:
            delta = "-"
        print(f"  {label:<16} {sx:>14} {sy:>14}   {delta:>8}")

    print(f"  {'':<16} {'A':>14} {'B':>14}   {'B vs A':>8}")
    row("triangles", a["triangles"], b["triangles"])
    for i, axis in enumerate("xyz"):
        row(f"bbox {axis} (mm)", a["bbox_mm"][i], b["bbox_mm"][i])
    row("volume (cm3)", a["volume_mm3"] / 1000, b["volume_mm3"] / 1000)
    row("area (cm2)", a["surface_area_mm2"] / 100, b["surface_area_mm2"] / 100)
    row("roughness (deg)", a["roughness_deg"], b["roughness_deg"])

    print("\n  Read roughness as detail retention: if B is much lower than A,")
    print("  the generative pass smoothed away ornament. Compare only after")
    print("  scaling both to the same longest dimension.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Check and compare STL files.")
    ap.add_argument("stl", type=Path)
    ap.add_argument("--compare", type=Path, metavar="OTHER.stl")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    try:
        a = analyse(args.stl)
        b = analyse(args.compare) if args.compare else None
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps({"a": a, "b": b} if b else a, indent=2))
    else:
        show(a)
        if b:
            show(b)
            compare(a, b)

    ok = a["printable"] and (b["printable"] if b else True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
