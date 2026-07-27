# Relief Forge Tools — Mesh Reconstruction and Fabrication Validation

Relief Forge Tools converts single-view or multi-view object photographs into
dimensionally scaled STL meshes and verifies that the resulting geometry is
suitable for fabrication. It targets small-scale investment casting and
additive manufacturing, where a model must be a closed solid with real
millimetre dimensions.

The reconstruction model is one interchangeable mesh source. The content of
this repository is the geometry processing and verification pipeline around it,
which operates identically on meshes from a generative model, a 3D scanner or a
CAD export.

## Pipeline

```
Photographs -> mesh generation -> physical scaling -> topology validation
            -> optional symmetry reconstruction -> STL candidate
```

Only generation requires PyTorch, model weights or a GPU. Every other stage
depends on NumPy alone. See `figures/generated/pipeline.pdf`.

## Core tools

**Mesh generation** (`cli/generate.py`) — single-view and multi-view input,
background removal, Hunyuan3D inference behind a pluggable backend, millimetre
scaling, face-count control, content-hash caching.

**Mesh validation** (`cli/mesh_check.py`) — binary and ASCII STL parsing,
vertex welding, boundary-edge detection, non-manifold-edge detection,
inconsistent-winding detection, connected-component analysis, signed-volume and
surface-area calculation, material mass estimation, geometry comparison.

**Symmetry reconstruction** (`cli/symmetrize.py`) — planar clipping, half-mesh
selection, reflection with winding reversal, removal of coplanar interior faces
and degenerate triangles, manifold revalidation. Appropriate only when
bilateral symmetry is a known property of the object; otherwise it discards
real asymmetric geometry.

## Example commands

```bash
python cli/mesh_check.py part.stl                     # validate and measure
python cli/mesh_check.py a.stl --compare b.stl        # compare two meshes
python cli/mesh_check.py part.stl --json              # machine-readable
python cli/symmetrize.py part.stl --axis x -o out.stl
python cli/generate.py photo.jpg --size 30 --resolution 384
python cli/generate.py front.jpg --views back.jpg left.jpg right.jpg
```

Exit codes: `0` pass, `1` checks failed, `2` usage or input error.

## Validation

A mesh **passes the repository's topological and orientation checks** when it
has no boundary edges, no non-manifold edges, no degenerate edges, consistent
local winding, positive signed volume, and the expected connected-component
count.

Passing establishes that the mesh is a closed, coherently oriented solid. It
does **not** validate minimum wall thickness, unsupported overhangs, printer
tolerances, casting shrinkage, trapped volumes, or any process-specific
constraint.

Tests use synthetic geometry with closed-form expected values: closed cube,
ASCII cube, inverted cube, missing face, reversed triangle, disconnected
solids, non-manifold flap, annulus, centred and offset symmetry cuts,
plane-intersection edge cases, zero-area triangle removal. Tolerances are
explicit per case — `1e-9` where exact in floating point, `3e-3` where a
polygonal approximation has a known discretisation error.

```bash
pytest                            # 71 tests, no GPU or network
python scripts/benchmark.py       # measured timings
python figures/make_figures.py    # regenerate figures
```

Details in `docs/methodology.tex` and `docs/validation.tex`.

## Hardware acceleration

**CPU-only.** Topology validation, STL analysis, symmetry reconstruction,
measurement and reporting need no GPU.

**NVIDIA CUDA (optional).** Device selection resolves at run time and falls
back automatically; CUDA is never assumed.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

Install a CUDA build of PyTorch matching your driver (CUDA 11.8 or 12.1 are the
usual wheel targets) and verify with `cli/generate.py --selftest`. Select a
device with `--device cuda|mps|cpu`; an unavailable explicit request raises
rather than silently downgrading.

**Higher-capacity models.** Discrete NVIDIA GPUs (RTX 3080/3090/4080/4090,
A5000, A6000) can run higher-fidelity checkpoints. These are not included here;
`MeshGenerator` is the extension point.

| Tier | Backend | Checkpoint | Reported VRAM |
|---|---|---|---|
| Default, lightweight | `hunyuan3d` | `default` | ~6 GB |
| Higher-capacity | `hunyuan3d-large` | `standard` | ~6 GB |
| Higher-fidelity | `hunyuan3d-large` | `large` | ~10 GB |

```yaml
model:
  backend: hunyuan3d-large
  checkpoint: large
  device: cuda
  fp16: true
```

VRAM figures are from upstream documentation, not measured here.

Geometry pipeline, measured on Linux aarch64, Python 3.10.12, NumPy 2.2.6,
median of five runs:

| Triangles | Parse | Weld | Topology | Peak memory |
|---|---|---|---|---|
| 512 | 0.01 ms | 0.42 ms | 3.03 ms | 0.21 MB |
| 8,192 | 0.04 ms | 6.62 ms | 50.7 ms | 2.96 MB |
| 32,768 | 0.18 ms | 27.5 ms | 208 ms | 11.8 MB |

No GPU inference benchmarks are reported; none have been measured.

## Limitations

Self-intersection is not detected. Welding tolerance is heuristic (`1e-6` of
the bounding-box diagonal). Mean dihedral angle is a limited detail-retention
proxy for closely related meshes at identical scale, not a perceptual quality
metric. Material mass assumes a fully dense solid, excluding sprue, flashing
and shrinkage. Single-view reconstruction infers unobserved surfaces rather
than measuring them.

## Installation

```bash
git clone https://github.com/tlosanti/relief-forge-tools.git
cd relief-forge-tools
pip install -e .        # geometry core: NumPy only
./scripts/setup.sh      # optional, generation dependencies
```

## License

MIT. Model weights are not distributed here. Hunyuan3D checkpoints downloaded
by the generation backend carry the Tencent Hunyuan licence, whose defined
territory excludes the EU, UK and South Korea and which restricts commercial
use. Review those terms before commercial application. See `LICENSE`.
