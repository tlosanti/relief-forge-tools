#!/usr/bin/env python3
"""Generate a dimensionally scaled STL mesh from one or more photographs.

This command is the only part of the toolkit that requires PyTorch, model
weights or a GPU. Validation, measurement and symmetry reconstruction operate
on any STL from any source and have no such requirement.

Model weights are not distributed with this repository; backends download
third-party checkpoints on first use under those projects' licence terms.

Exit codes
----------
0
    Mesh generated and written.
1
    Generation failed.
2
    Usage or input error.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # type: ignore[import-not-found]
    EXIT_CHECKS_FAILED,
    EXIT_OK,
    EXIT_USAGE_ERROR,
    ensure_package_on_path,
    format_report,
)

ensure_package_on_path()

from relief_forge import ReliefForgeError, analyse_mesh, write_stl  # noqa: E402
from relief_forge.generation import (  # noqa: E402
    GenerationConfig,
    available_backends,
    get_backend,
    load_images,
    resolve_device,
    scale_to_millimetres,
)

DEFAULT_CACHE = Path("gen3d_cache")
DEFAULT_OUTPUT = Path.home() / "Desktop" / "Filter Exports"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate",
        description="Photograph to dimensionally scaled STL mesh.",
    )
    parser.add_argument("image", type=Path, nargs="?", help="primary view")
    parser.add_argument(
        "--views",
        type=Path,
        nargs="*",
        default=[],
        help="additional views. Multi-view input constrains geometry that a "
        "single view leaves ambiguous.",
    )
    parser.add_argument(
        "--backend",
        default="hunyuan3d",
        choices=list(available_backends()),
        help="mesh-generation backend (default: hunyuan3d)",
    )
    parser.add_argument(
        "--checkpoint",
        default="default",
        help="backend-specific checkpoint identifier (default: default)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="compute device; auto prefers CUDA, then MPS, then CPU",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="request half precision (honoured on CUDA only)",
    )
    parser.add_argument("--size", type=float, default=30.0, help="longest dimension in mm")
    parser.add_argument("--steps", type=int, default=30, help="denoising steps")
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="volume grid resolution; detail below one cell cannot be represented",
    )
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--max-faces", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--no-rembg", action="store_true", help="input is already background-free"
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="check the environment without loading a model",
    )
    return parser


def selftest() -> int:
    """Report environment readiness without downloading model weights."""
    print("relief-forge generate - environment check\n")
    ok = True

    def probe(label: str, function) -> None:  # type: ignore[no-untyped-def]
        nonlocal ok
        try:
            print(f"  [ok]   {label}: {function()}")
        except Exception as exc:
            ok = False
            print(f"  [FAIL] {label}: {type(exc).__name__}: {exc}")

    probe("python", lambda: sys.version.split()[0])
    probe("numpy", lambda: __import__("numpy").__version__)
    probe("relief_forge", lambda: __import__("relief_forge").__version__)
    probe("torch", lambda: __import__("torch").__version__)
    probe("device", lambda: resolve_device("auto"))
    probe("PIL", lambda: __import__("PIL").__version__)

    try:
        import hy3dgen  # noqa: F401

        print("  [ok]   hy3dgen importable")
    except ImportError as exc:
        ok = False
        print(f"  [FAIL] hy3dgen: {exc}")
        print("         Run scripts/setup.sh to install generation dependencies.")

    print(f"\n  backends registered: {', '.join(available_backends())}")
    print("\n" + ("environment ready." if ok else "environment incomplete."))
    return EXIT_OK if ok else EXIT_CHECKS_FAILED


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.selftest:
        return selftest()
    if args.image is None:
        print("error: an image is required (or use --selftest)", file=sys.stderr)
        return EXIT_USAGE_ERROR

    paths = [args.image, *args.views]
    for path in paths:
        if not path.exists():
            print(f"error: no such file: {path}", file=sys.stderr)
            return EXIT_USAGE_ERROR

    config = GenerationConfig(
        backend=args.backend,
        checkpoint=args.checkpoint,
        device=args.device,
        fp16=args.fp16,
        steps=args.steps,
        resolution=args.resolution,
        guidance=args.guidance,
        seed=args.seed,
        max_faces=args.max_faces,
        remove_background=not args.no_rembg,
        target_size_mm=args.size,
    )

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    key = config.cache_key(paths)
    cached = args.cache_dir / f"{key}.stl"

    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    destination_dir = args.output_dir / stamp
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{args.image.stem}_{config.backend}.stl"

    elapsed = 0.0
    cache_hit = cached.exists()

    if cache_hit:
        shutil.copy2(cached, destination)
    else:
        try:
            backend = get_backend(config)
            images = load_images(paths, config.remove_background)
            started = time.time()
            triangles = backend.generate(images)
            elapsed = time.time() - started
            scaled = scale_to_millimetres(triangles, config.target_size_mm)
            write_stl(cached, scaled)
            write_stl(destination, scaled)
        except ReliefForgeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_CHECKS_FAILED
        except OSError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_USAGE_ERROR

    try:
        report = analyse_mesh(destination)
    except ReliefForgeError as exc:
        print(f"error: generated mesh could not be analysed: {exc}", file=sys.stderr)
        return EXIT_CHECKS_FAILED

    if args.json:
        print(
            json.dumps(
                {
                    "output": str(destination),
                    "cache_key": key,
                    "cache_hit": cache_hit,
                    "seconds": round(elapsed, 3),
                    "config": {
                        "backend": config.backend,
                        "checkpoint": config.checkpoint,
                        "device": config.device,
                        "fp16": config.fp16,
                        "steps": config.steps,
                        "resolution": config.resolution,
                        "target_size_mm": config.target_size_mm,
                        "n_views": len(paths),
                    },
                    "report": report.to_dict(),
                },
                indent=2,
            )
        )
    else:
        print(f"{'cache hit' if cache_hit else f'generated in {elapsed:.1f} s'} ({key})")
        print(f"output: {destination}\n")
        print(format_report(report))

    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
