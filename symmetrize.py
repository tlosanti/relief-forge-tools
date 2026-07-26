#!/usr/bin/env python3
"""
symmetrize.py — cut a mesh at a plane, keep one side, mirror it.

The trick already used on the depth side, applied to a full 3D mesh instead of
a relief: keep whichever half read cleaner and mirror it to make the whole
piece exact rather than merely plausible.

Why this matters more for generative output than for depth. A model like
Hunyuan3D guesses the geometry it cannot see, and it guesses *differently* on
the left than on the right -- so a piece that is genuinely symmetric comes back
subtly lopsided. Cutting and mirroring replaces the model's worse guess with
its better one and enforces the symmetry as a hard constraint. It removes
error rather than averaging it.

Note the difference from the depth workflow: mirroring about X (left/right)
is a real property of a bilaterally symmetric piece and is safe. Mirroring
about Z (front/back) is not -- it fabricates a back that is a copy of the
front, which is why that step tends to disappoint on pendants whose real back
is flat or plain. With a generative mesh you already have a back, so the Z
mirror is no longer needed.

Pure numpy. No trimesh.

  python3 tools/symmetrize.py piece.stl -o out.stl            # keep x<=0
  python3 tools/symmetrize.py piece.stl --keep +x -o out.stl  # keep x>=0
  python3 tools/symmetrize.py piece.stl --axis y --at 1.5 -o out.stl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mesh_check import analyse, read_stl, show, write_stl  # noqa: E402

AXES = {"x": 0, "y": 1, "z": 2}


def clip_triangles(tris: np.ndarray, axis: int, at: float,
                   keep_positive: bool) -> np.ndarray:
    """Sutherland-Hodgman clip of each triangle against a plane.

    Returns the kept portion, retriangulated. Triangles straddling the plane
    produce a quad (-> 2 tris) or a triangle (-> 1). Winding is preserved, so
    a consistently wound input stays consistently wound.

    The cut leaves an open boundary lying exactly on the plane. That is
    intentional: the mirrored copy has the identical boundary, so welding the
    two closes the seam with no cap geometry and no cracks.

    Three degenerate cases have to be handled or the mirrored result is not
    manifold. All three occur in practice whenever the mesh happens to have
    vertices sitting on the cut plane, which is common on a symmetric piece:

      * a triangle lying entirely *in* the plane would be duplicated by its
        own mirror image, producing a doubled interior wall -> dropped
      * a triangle on the far side touching the plane at one vertex clips to
        a zero-area sliver -> dropped
      * a crossing computed at a vertex already on the plane repeats that
        point in the polygon -> deduplicated before triangulating
    """
    flat = tris.reshape(-1, 3)
    diag = float(np.linalg.norm(flat.max(axis=0) - flat.min(axis=0)))
    eps = diag * 1e-7
    area_eps = (diag * 1e-6) ** 2

    sign = 1.0 if keep_positive else -1.0
    out = []

    for tri in tris:
        d = sign * (tri[:, axis] - at)
        d = np.where(np.abs(d) < eps, 0.0, d)  # snap near-plane to on-plane

        if np.all(d == 0.0):
            continue  # coplanar with the cut; the mirror would duplicate it

        inside = d >= 0.0
        n_in = int(inside.sum())
        if n_in == 0:
            continue
        if n_in == 3:
            out.append(tri)
            continue

        # Walk the triangle's edges, emitting kept vertices and crossings.
        poly: list[np.ndarray] = []

        def push(p: np.ndarray) -> None:
            if poly and np.linalg.norm(p - poly[-1]) < eps:
                return
            poly.append(p)

        for i in range(3):
            j = (i + 1) % 3
            if inside[i]:
                push(tri[i])
            # A vertex exactly on the plane is already emitted above; only a
            # true sign change generates a new crossing point.
            if (d[i] > 0) != (d[j] > 0) and d[i] != 0.0 and d[j] != 0.0:
                t = d[i] / (d[i] - d[j])
                p = tri[i] + t * (tri[j] - tri[i])
                p[axis] = at  # snap exactly onto the plane
                push(p)

        if len(poly) > 1 and np.linalg.norm(poly[0] - poly[-1]) < eps:
            poly.pop()
        if len(poly) < 3:
            continue

        poly_a = np.asarray(poly)
        for k in range(1, len(poly_a) - 1):  # fan triangulate
            out.append(np.stack([poly_a[0], poly_a[k], poly_a[k + 1]]))

    if not out:
        return np.zeros((0, 3, 3))

    res = np.asarray(out)
    a, b, c = res[:, 0], res[:, 1], res[:, 2]
    area2 = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2.0
    return res[area2 > area_eps]


def mirror(tris: np.ndarray, axis: int, at: float) -> np.ndarray:
    """Reflect across the plane and reverse winding.

    Reflection flips handedness, so without the winding reversal every normal
    on the mirrored half would point inward and the result would fail the
    signed-volume test.
    """
    m = tris.copy()
    m[:, :, axis] = 2.0 * at - m[:, :, axis]
    return m[:, ::-1, :]


def symmetrize(tris: np.ndarray, axis: int, at: float,
               keep_positive: bool) -> np.ndarray:
    half = clip_triangles(tris, axis, at, keep_positive)
    if len(half) == 0:
        raise ValueError(
            f"nothing left after cutting at {'xyz'[axis]}={at}. "
            "Check the plane position — the mesh may sit entirely on one side.")
    return np.concatenate([half, mirror(half, axis, at)])


def main() -> int:
    ap = argparse.ArgumentParser(description="Mirror half a mesh onto itself.")
    ap.add_argument("stl", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--axis", choices=list(AXES), default="x",
                    help="mirror plane normal (default x = left/right)")
    ap.add_argument("--at", type=float, default=None,
                    help="plane position; default is the bounding-box centre")
    ap.add_argument("--keep", choices=["-x", "+x"], default="-x",
                    help="which side survives (default -x)")
    args = ap.parse_args()

    try:
        tris = read_stl(args.stl)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    axis = AXES[args.axis]
    if args.at is None:
        flat = tris.reshape(-1, 3)
        at = float((flat[:, axis].min() + flat[:, axis].max()) / 2.0)
        print(f"plane: {args.axis} = {at:.4f} (bounding-box centre)")
    else:
        at = args.at
        print(f"plane: {args.axis} = {at:.4f}")

    try:
        result = symmetrize(tris, axis, at, keep_positive=args.keep == "+x")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_stl(args.out, result)
    print(f"{len(tris):,} triangles in -> {len(result):,} out")

    r = analyse(args.out)
    show(r)
    if not r["printable"]:
        print("\n  The result is not printable. If the input was watertight,")
        print("  the usual cause is a plane that grazes coplanar geometry.")
        print("  Nudge --at slightly and retry.")
    return 0 if r["printable"] else 1


if __name__ == "__main__":
    sys.exit(main())
