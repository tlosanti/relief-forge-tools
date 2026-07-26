# relief-forge-tools

Turn a photograph of an object into a watertight STL you can print or cast.

Built for jewellery: small pieces, cast in silver or bronze, where the model
has to be a closed solid and the wall thicknesses have to be real millimetres.

```bash
./setup_gen3d.sh
.venv-gen3d/bin/python gen3d.py pendant.jpg --size 30
python3 mesh_check.py "path/it/prints.stl"
```

## What each script does

| Script | Purpose | Needs |
|---|---|---|
| `gen3d.py` | photo → closed 3D mesh → millimetre-scaled STL | GPU, ~8 GB model download |
| `mesh_check.py` | is this STL actually printable? | numpy only |
| `symmetrize.py` | keep half a mesh, mirror it onto itself | numpy only |
| `setup_gen3d.sh` | one-time install | git, internet |

`mesh_check.py` and `symmetrize.py` are standalone. They work on any STL from
any source — Blender, a scanner, another generator — and have no dependency on
the generation half of this repo.

## Install

```bash
git clone <this repo>
cd relief-forge-tools
chmod +x setup_gen3d.sh
./setup_gen3d.sh
.venv-gen3d/bin/python gen3d.py --selftest
```

Run `--selftest` before anything else. It checks the environment in a second
or two, rather than letting you discover a problem after an 8 GB download.

The installer picks its own mode. If the sibling Half-Tone-Depth-Wrap app is
present it reuses that venv's torch; otherwise it builds a self-contained
environment and installs torch itself. Either way it never writes to anything
outside this folder.

Confirmed working on Apple Silicon, Python 3.14, torch 2.13, MPS.

## Generate

```bash
.venv-gen3d/bin/python gen3d.py pendant.jpg --size 30
```

`--size` is the longest dimension in millimetres. Output lands in
`~/Desktop/Filter Exports/<timestamp>/`.

The model download is ~7.7 GB, once per machine, cached in `~/.cache/hy3dgen`.
Results are then cached per photo by content hash, so re-running the same image
at the same settings costs nothing.

**Multi-view is the single biggest quality lever**, and needs no training:

```bash
.venv-gen3d/bin/python gen3d.py front.jpg --views back.jpg left.jpg right.jpg
```

One photo means the model invents everything it cannot see. Four photos means
it mostly interpolates. If you can shoot the piece from several angles, do.

Useful flags: `--octree 384` for finer detail (slower), `--steps`, `--no-rembg`
if the image is already cut out, `--faces` to cap triangle count.

## Check before you print

```bash
python3 mesh_check.py piece.stl
```

Asserts the two things that decide whether an STL is printable: every directed
edge has exactly one opposite twin (watertight), and total signed volume is
positive (normals face outward). Also reports bounding box, triangle count,
connected components, and **cast weight in silver, bronze, 18k gold and resin**
— which is the number that tells you what a piece will cost to pour.

Exit code is 0 only if the mesh passes, so it can gate a pipeline.

Compare two versions of the same object:

```bash
python3 mesh_check.py a.stl --compare b.stl
```

The `roughness` figure (mean dihedral angle) acts as a detail-retention proxy.
If a generated mesh scores much lower than a reference, fine surface ornament
has been smoothed away. Only compare meshes scaled to the same longest
dimension.

## Symmetry

```bash
python3 symmetrize.py piece.stl --at 0 -o final.stl
```

Cuts at a plane, keeps one side, mirrors it. On a bilaterally symmetric piece
this is not an approximation — it is a true fact about the object, and it
strictly removes error: you keep the half that came out clean and discard the
half that did not.

It is especially worth doing on generated meshes. The model's guesses for the
left and right sides always differ slightly, so mirroring replaces its worse
guess with its better one.

Mirroring front-to-back is a different matter and generally a mistake: it
fabricates a reverse side that is a copy of the front, when most cast pendants
have a plain or flat back. That information was never in the photograph.

## Licence

Code here is MIT — do what you like with it.

**The model is not.** These scripts download and run Tencent's Hunyuan3D,
which carries its own [Non-Commercial License
Agreement](https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/LICENSE).
Its defined territory excludes the EU, UK and South Korea, and published
summaries disagree about what commercial use it allows. If you intend to sell
what you make, read those terms yourself. See `LICENSE` for detail. Not legal
advice.

## Known limits

- Monocular input gives a *relief-like* result: the visible face is faithful,
  the unseen side is inferred. Multi-view fixes most of this.
- Fine ornament — filigree, beading, engraved lines — tends to soften. Use
  `--octree 384` and avoid aggressive `--faces` reduction if detail matters
  more than file size. You can grind material off a wax; you cannot grind it
  back on.
- Texture generation is CUDA-only and deliberately not attempted. Geometry is
  all that matters for casting.

For design notes, verification details and troubleshooting, see
[README-gen3d.md](README-gen3d.md).
