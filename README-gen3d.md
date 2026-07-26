# tools/ — generative 3D experiment

An alternative to the depth pipeline: a generative model that produces a full
closed mesh instead of a relief. Kept separate from the app on purpose —
nothing here touches `server.py`, `static/main.js`, or the shared `.venv`, so
Relief Forge cannot be broken by anything in this folder.

| File | Runs on | Needs |
|---|---|---|
| `mesh_check.py` | anything | numpy only |
| `symmetrize.py` | anything | numpy only |
| `gen3d.py` | your Mac | `.venv-gen3d`, ~2 GB model download |
| `setup_gen3d.sh` | your Mac | git, internet |

## Setup

```bash
cd ~/Desktop/relief-forge
chmod +x tools/setup_gen3d.sh
./tools/setup_gen3d.sh
.venv-gen3d/bin/python tools/gen3d.py --selftest
```

`setup_gen3d.sh` creates `.venv-gen3d` and points it at the halftone app's
`site-packages` with a `.pth` file, so torch is inherited rather than
reinstalled — the same "one large install" rule `run.sh` follows. The shared
venv is never written to.

A `.pth` is used instead of `venv --system-site-packages` because a venv
created from another venv inherits the *base* interpreter's packages, not the
parent venv's. That distinction bites silently, hence the note.

## Generate

```bash
.venv-gen3d/bin/python tools/gen3d.py pendant.jpg --size 30
```

Output lands in `~/Desktop/Filter Exports/<timestamp>/` alongside your normal
exports, and caches by content hash in `gen3d_cache/` — same rule as
`depth_cache/`, so re-running a photo costs nothing.

Multi-view is the single biggest quality lever, and it needs no training:

```bash
.venv-gen3d/bin/python tools/gen3d.py front.jpg --views back.jpg left.jpg right.jpg
```

Each additional angle removes guesswork. One photo means the model invents
everything it cannot see; four photos means it mostly interpolates.

## Check

```bash
python3 tools/mesh_check.py piece.stl
python3 tools/mesh_check.py forge.stl --compare hunyuan.stl
```

Same two assertions the existing STL code is held to — every directed edge has
exactly one opposite twin, and total signed volume is positive — plus bbox,
cast weights, component count, and a `roughness` figure (mean dihedral angle)
that acts as a **detail-retention proxy**. If the generative mesh comes back
with much lower roughness than the Relief Forge STL of the same photo, it
smoothed away the ornament. Only compare after scaling both to the same
longest dimension. Exit code is 0 only if watertight and outward-wound, so it
can gate a pipeline.

## Symmetry

```bash
python3 tools/symmetrize.py hunyuan.stl --at 0 -o final.stl
```

This is your delete-half-and-mirror trick, applied to a full mesh rather than
a relief — and it is *more* useful here than it was on the depth side.

The reason it underperformed before is worth separating into two halves,
because one half of the idea is sound and the other is not:

- **Mirroring left/right is sound.** On a bilaterally symmetric piece it is a
  true fact about the object. It is not an approximation, and it strictly
  removes error: you keep the half that read cleanly and discard the half that
  did not.
- **Mirroring front/back is where it fell down.** `symZ` fabricates a back
  that is a copy of the front. Most cast pendants have a plain or flat
  reverse, so you get a piece that is too thick and wrong on the side you
  cannot see. No amount of tuning fixes that, because the information was
  never in the photograph.

A generative mesh removes the second problem — it gives you an actual back —
while leaving the first advantage fully intact. So the ordering that makes
sense is: generate, then cut and mirror. The model's left-side and right-side
guesses always differ slightly, and mirroring replaces its worse guess with
its better one.

`symmetrize.py` snaps near-plane vertices, drops faces lying in the cut plane
(their mirror would duplicate them into an interior wall), and discards
zero-area slivers. Without those three, any mesh with vertices on the cut
plane comes out non-manifold — which is exactly what happened on the first
version, caught by the annulus test.

## What is verified, and what is not

Verified here, against synthetic meshes with known analytic answers:

- `mesh_check.py` returns the correct verdict on a cube, an ASCII-encoded
  cube, an inverted cube, a cube with a missing face, a cube with one
  reversed triangle, two disjoint solids, a non-manifold flap, and an annulus.
  The cube's mean dihedral lands on exactly 60° as its geometry predicts
  (12 edges at 90°, 6 face diagonals at 0°); annulus volume reads 790 mm³
  against an analytic 791.7 mm³, the gap being the 64-gon approximation.
- `symmetrize.py` reproduces a cube exactly when cut through its own centre,
  turns a −10..+5 box into a watertight 20×12×4 at exactly 0.960 cm³, and
  preserves the annulus and its hole. The offset-sphere case matches the
  spherical-cap formula to within the test mesh's own faceting error.

- `accepted()` in `gen3d.py`, the keyword filter, behaves correctly against a
  full signature, an older signature missing arguments, a `**kwargs` passthrough,
  a bound method, a `functools.wraps` decorated function, and a builtin whose
  signature is misleading enough that filtering must be abandoned entirely.

**Partly verified:** `setup_gen3d.sh` and `--selftest` are confirmed working
on this machine — Python 3.14.6 on both sides of the link, torch 2.13.0
inherited rather than reinstalled, MPS live, `hy3dgen` importable.

**Not verified:** the generation path in `gen3d.py` has never been executed.
The environment it was written in has no GPU, 3 GB of RAM, and no HuggingFace
access. The pipeline call and mesh conversion come from documentation, not
from a successful run. See Troubleshooting for the failures anticipated and
guarded against.

The texture and rasterizer extensions are CUDA-only and will fail to build on
Apple Silicon. That is expected and harmless — shape generation does not use
them, and geometry is all that matters for casting.

## Troubleshooting

Setup is confirmed working on Python 3.14 / torch 2.13 / MPS. Everything below
concerns the generation run itself, which is the part that had never been
executed when it was written.

**Nothing prints for a long time on first run.** Expected. The model is
~7.7 GB into `~/.cache/hy3dgen`. Once per machine, not per photo.

**`TypeError: __call__() got an unexpected keyword argument ...`**
Should no longer happen — arguments are filtered against the real signature
first, and anything this build doesn't accept prints as a `[note]`. If it
still occurs, the signature is being masked by a wrapper; paste the error.

**`ModuleNotFoundError: diso`, or an error mentioning DMC / marching cubes.**
The default surface extractor is CUDA-only. Plain marching cubes is requested
up front on non-CUDA devices, with a retry as backup. If both fail, the build
has no CPU extractor and only a newer hy3dgen will help.

**`NotImplementedError: ... not implemented for MPS`.**
`PYTORCH_ENABLE_MPS_FALLBACK=1` is set at the top of `gen3d.py` before torch
loads, which routes missing kernels to CPU. If one still escapes, force the
whole run onto CPU — slow but reliable:

```bash
PYTORCH_MPS_DISABLE=1 .venv-gen3d/bin/python tools/gen3d.py photo.png --size 30
```

**Runs out of memory.** Drop `--octree 256` to `192`, or `--steps` to `20`.

**Produces a blob with no recognisable shape.** Usually background removal
took the wrong thing. Check what it saw by cutting the image out yourself and
passing `--no-rembg`.

**Mesh comes out inside-out** (`mesh_check` says "outward wound NO"). Known
possibility with generative output; the geometry is fine, the winding is
reversed. Worth reporting — it is a two-line fix in the exporter, not a
regeneration.

## Expected outcome on the Quimbaya pendant

Worth setting expectations before spending an evening on it. The filigree
spirals on the wings and the beading along the tail are precisely the scale of
detail these models smooth into blobs. The likely result is a good silhouette
and a real back, with the ornament softened — the opposite failure to the
depth pipeline, which keeps surface detail faithfully and cannot see the back
at all.

`--compare` is there to turn that from an impression into a number. If it
holds, the interesting build is not to replace depth but to use the generative
mesh for the body and the depth map for the front surface.

Smoothing the wax with a die grinder afterwards changes this calculus in one
specific way: it makes *lost* detail expensive and *excess* detail cheap. You
can grind material away, but you cannot grind ornament back on. That argues
for the higher `--octree 384` setting and against aggressive `--faces`
reduction, even though both cost time.
