#!/usr/bin/env python3
"""
gen3d.py — Hunyuan3D shape generation for Relief Forge, Apple Silicon.

Photo -> full closed 3D mesh -> millimetre-scaled STL. Unlike the depth
pipeline this produces a real back side, but it *invents* the geometry the
camera never saw, and it smooths fine surface ornament. Treat the output as a
volume proposal, not as a faithful record of the object.

Shape only. Texture generation is CUDA-only and is deliberately not attempted;
for casting and printing the geometry is all that matters anyway.

Follows the project caching rule: results key on a content hash of the input
image plus the generation parameters, so re-running the same photo with the
same settings is instant.

  .venv-gen3d/bin/python tools/gen3d.py --selftest
  .venv-gen3d/bin/python tools/gen3d.py pendant.jpg --size 30
  .venv-gen3d/bin/python tools/gen3d.py pendant.jpg --views back.jpg side.jpg

Heavy imports are deferred so --help and --selftest stay fast.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# MPS lacks kernels for a handful of ops the model uses. Without this the run
# dies partway with "not implemented for MPS"; with it, those ops fall back to
# CPU and everything completes. Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Works both as relief-forge/tools/gen3d.py and as the root of a standalone
# checkout, so the repo is usable by someone who does not have the rest of the
# Relief Forge apps on their machine.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == "tools" else HERE
CACHE = ROOT / "gen3d_cache"
OUT = Path.home() / "Desktop" / "Filter Exports"


def shared_venv_python() -> Path | None:
    """The sibling halftone app's interpreter, if this is a Relief Forge
    checkout. Returns None on a standalone install, which is not an error."""
    p = ROOT.parent / "Half-Tone-Depth-Wrap" / ".venv" / "bin" / "python"
    return p if p.exists() else None

REPO = "tencent/Hunyuan3D-2mini"
SUBFOLDER = "hunyuan3d-dit-v2-mini-turbo"


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------

def pick_device():
    import torch

    if torch.backends.mps.is_available():
        return "mps", "Apple GPU (MPS)"
    if torch.cuda.is_available():
        return "cuda", torch.cuda.get_device_name(0)
    return "cpu", "CPU only — expect this to take many minutes"


def selftest() -> int:
    """Check the install before committing to a multi-GB download."""
    print("Relief Forge — gen3d selftest\n")
    ok = True

    def probe(label: str, fn):
        nonlocal ok
        try:
            print(f"  [ok]   {label}: {fn()}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"  [FAIL] {label}: {type(exc).__name__}: {exc}")

    probe("python", lambda: sys.version.split()[0])

    # The shared-venv link is version-locked: torch's compiled extensions only
    # load under the exact Python minor they were built for. A mismatch here is
    # the difference between "torch is missing" and "torch cannot ever load".
    shared = shared_venv_python()
    if shared is not None:
        import subprocess
        try:
            theirs = subprocess.run(
                [str(shared), "-c", 'import sys;print("%d.%d"%sys.version_info[:2])'],
                capture_output=True, text=True, timeout=30).stdout.strip()
            mine = "%d.%d" % sys.version_info[:2]
            if theirs == mine:
                print(f"  [ok]   shared venv python: {theirs} (matches)")
            else:
                ok = False
                print(f"  [FAIL] python mismatch: this venv is {mine}, "
                      f"shared venv is {theirs}")
                print("         Compiled packages cannot cross versions. Fix:")
                print("           rm -rf .venv-gen3d && ./tools/setup_gen3d.sh")
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] could not query shared venv: {exc}")
    else:
        print("  [ok]   standalone install (no sibling venv to match against)")

    probe("numpy", lambda: __import__("numpy").__version__)
    probe("torch", lambda: __import__("torch").__version__)
    probe("trimesh", lambda: __import__("trimesh").__version__)
    probe("PIL", lambda: __import__("PIL").__version__)

    try:
        import torch  # noqa: F401
        dev, desc = pick_device()
        print(f"  [ok]   device: {dev} — {desc}")
        if dev == "cpu":
            print("         (no GPU found; generation will be very slow)")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  [FAIL] device: {exc}")

    try:
        import hy3dgen  # noqa: F401
        print("  [ok]   hy3dgen importable")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  [FAIL] hy3dgen: {exc}")
        print("         Run ./tools/setup_gen3d.sh first.")

    try:
        from hy3dgen.rembg import BackgroundRemover  # noqa: F401
        print("  [ok]   background remover available")
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] background remover unavailable ({exc});")
        print("         pass --no-rembg and pre-cut the photo yourself.")

    print("\n" + ("selftest passed." if ok else "selftest FAILED — see above."))
    return 0 if ok else 1


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------

def cache_key(images: list[Path], args) -> str:
    h = hashlib.sha256()
    for p in images:
        h.update(p.read_bytes())
    h.update(json.dumps({
        "repo": REPO, "sub": SUBFOLDER, "steps": args.steps,
        "octree": args.octree, "guidance": args.guidance, "seed": args.seed,
        "n": len(images),
    }, sort_keys=True).encode())
    return h.hexdigest()[:16]


def load_images(paths: list[Path], use_rembg: bool):
    from PIL import Image

    imgs = []
    for p in paths:
        im = Image.open(p).convert("RGBA")
        if use_rembg:
            from hy3dgen.rembg import BackgroundRemover
            im = BackgroundRemover()(im.convert("RGB"))
        imgs.append(im)
    return imgs


def accepted(fn, kwargs: dict) -> dict:
    """Drop keyword arguments the callable does not accept.

    hy3dgen's signatures drift between releases, and passing one stale keyword
    raises TypeError after the model is already loaded -- the most expensive
    possible moment to fail. Filtering against the real signature turns that
    into a printed note and a working run.
    """
    import inspect

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return kwargs  # can't introspect; hope for the best

    if any(p.kind is inspect.Parameter.VAR_KEYWORD
           for p in sig.parameters.values()):
        return kwargs

    keep = {k: v for k, v in kwargs.items() if k in sig.parameters}

    # If the essential argument vanished, the signature we read is not the one
    # that will actually be called (wrappers and decorators do this). Filtering
    # on a misleading signature is worse than not filtering at all.
    if "image" in kwargs and "image" not in keep:
        print("    [note] signature looks wrapped; passing arguments through")
        return kwargs

    dropped = set(kwargs) - set(keep)
    if dropped:
        print(f"    [note] this hy3dgen build ignores: {', '.join(sorted(dropped))}")
    return keep


def load_pipeline(dev: str):
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline as Pipe

    print(f"    loading {REPO}/{SUBFOLDER}")
    print("    (first run downloads ~7.7 GB to ~/.cache/hy3dgen — once per machine)")

    kw = accepted(Pipe.from_pretrained,
                  {"subfolder": SUBFOLDER, "use_safetensors": True, "device": dev})
    pipe = Pipe.from_pretrained(REPO, **kw)

    # If the loader ignored `device`, move the pipeline across by hand.
    if "device" not in kw:
        for attr in ("to", "cuda"):
            mover = getattr(pipe, attr, None)
            if attr == "to" and callable(mover):
                try:
                    pipe = mover(dev)
                except Exception as exc:  # noqa: BLE001
                    print(f"    [warn] could not move pipeline to {dev}: {exc}")
                break
    return pipe


def generate(images, args):
    import torch

    dev, desc = pick_device()
    print(f"    device: {dev} ({desc})")

    pipe = load_pipeline(dev)

    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    payload = images[0] if len(images) == 1 else images

    base = {
        "image": payload,
        "num_inference_steps": args.steps,
        "octree_resolution": args.octree,
        "guidance_scale": args.guidance,
        "generator": gen,
    }

    # Surface extraction: the default differentiable marching cubes ("dmc")
    # depends on the `diso` package, which is CUDA-only. On Apple Silicon it
    # fails *after* the whole diffusion run has finished -- so on non-CUDA
    # devices ask for plain marching cubes up front, and keep the swap as a
    # retry in case the keyword is spelled differently in this build.
    attempts = [base] if dev == "cuda" else [
        {**base, "mc_algo": "mc"},
        base,
    ]

    print(f"    generating: {args.steps} steps, octree {args.octree}")
    t0 = time.time()
    last: Exception | None = None

    for i, kwargs in enumerate(attempts):
        try:
            mesh = pipe(**accepted(pipe.__call__, kwargs))[0]
            print(f"    done in {time.time() - t0:.1f}s")
            if not args.raw:
                mesh = clean(mesh, args.faces)
            return mesh
        except Exception as exc:  # noqa: BLE001
            last = exc
            blob = f"{type(exc).__name__}: {exc}".lower()
            cuda_ish = any(w in blob for w in
                           ("diso", "dmc", "cuda", "marching", "not implemented"))
            if i + 1 < len(attempts) and cuda_ish:
                print(f"    [retry] {type(exc).__name__}: {exc}")
                print("    [retry] falling back to plain marching cubes")
                continue
            raise

    raise RuntimeError(f"generation failed: {last}")


def clean(mesh, target_faces: int):
    """Strip debris and cap triangle count. Failures here are non-fatal."""
    from hy3dgen.shapegen import (DegenerateFaceRemover, FaceReducer,
                                  FloaterRemover)

    for label, step in (
        ("floaters", FloaterRemover()),
        ("degenerate faces", DegenerateFaceRemover()),
    ):
        try:
            mesh = step(mesh)
        except Exception as exc:  # noqa: BLE001
            print(f"    [warn] {label} cleanup skipped: {exc}")

    try:
        mesh = FaceReducer()(mesh, max_facenum=target_faces)
    except Exception as exc:  # noqa: BLE001
        print(f"    [warn] face reduction skipped: {exc}")

    return mesh


def to_trimesh(mesh):
    import numpy as np
    import trimesh

    if isinstance(mesh, trimesh.Trimesh):
        return mesh
    verts = np.asarray(getattr(mesh, "vertices", getattr(mesh, "verts", None)))
    faces = np.asarray(getattr(mesh, "faces", None))
    if verts is None or faces is None:
        raise TypeError(f"cannot convert {type(mesh)} to a mesh")
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def scale_to_mm(tm, longest_mm: float):
    """Hunyuan emits a unit-ish box. Scale so the longest axis is longest_mm,
    then sit the piece on z=0 and centre it in x/y for printing."""
    extents = tm.bounds[1] - tm.bounds[0]
    longest = float(extents.max())
    if longest <= 0:
        raise ValueError("mesh has zero extent")

    tm.apply_scale(longest_mm / longest)
    lo, hi = tm.bounds
    tm.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    return tm


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Photo -> closed 3D mesh -> STL (Hunyuan3D, shape only).")
    ap.add_argument("image", type=Path, nargs="?", help="front view photo")
    ap.add_argument("--views", type=Path, nargs="*", default=[],
                    help="extra angles (back, sides). Multi-view removes most "
                         "of the guessing and is a large quality win.")
    ap.add_argument("--size", type=float, default=30.0,
                    help="longest dimension in mm (default 30)")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--octree", type=int, default=256,
                    help="volume resolution; 384 is finer but much slower")
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--faces", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--raw", action="store_true", help="skip mesh cleanup")
    ap.add_argument("--no-rembg", action="store_true",
                    help="input is already cut out")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not args.image:
        ap.error("an image is required (or use --selftest)")

    paths = [args.image] + list(args.views)
    for p in paths:
        if not p.exists():
            print(f"error: no such file: {p}", file=sys.stderr)
            return 2

    CACHE.mkdir(exist_ok=True)
    key = cache_key(paths, args)
    cached = CACHE / f"{key}.stl"

    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    dest_dir = OUT / stamp
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{args.image.stem}_hunyuan.stl"

    if cached.exists():
        print(f"==> Cache hit ({key}) — no inference needed")
        import shutil
        shutil.copy2(cached, dest)
    else:
        print(f"==> Generating ({len(paths)} view{'s' if len(paths) > 1 else ''})")
        try:
            imgs = load_images(paths, use_rembg=not args.no_rembg)
            mesh = generate(imgs, args)
            tm = scale_to_mm(to_trimesh(mesh), args.size)
        except ImportError as exc:
            print(f"\nerror: {exc}\nRun ./tools/setup_gen3d.sh first.",
                  file=sys.stderr)
            return 2
        except Exception as exc:  # noqa: BLE001
            print(f"\nerror during generation: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            return 1

        tm.export(cached)
        tm.export(dest)
        print(f"    cached as {key}")

    print(f"\n==> {dest}")
    print("\nNow check it:")
    print(f"  python3 tools/mesh_check.py '{dest}'")
    print("  python3 tools/mesh_check.py forge.stl --compare "
          f"'{dest}'   # detail retention")

    if sys.platform == "darwin":
        os.system(f'open "{dest_dir}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
