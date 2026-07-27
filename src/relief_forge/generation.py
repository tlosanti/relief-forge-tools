"""Backend-agnostic image-to-mesh generation.

The geometry pipeline in this package operates on triangle arrays and has no
knowledge of how a mesh was produced. This module is the only place that
imports a reconstruction model, and it exposes a single interface,
:class:`MeshGenerator`, so that adding a new model requires implementing one
subclass and registering it. Downstream scaling, topology analysis, symmetry
reconstruction and STL writing are unchanged by that addition.

No model weights are distributed with this package. Backends download
third-party checkpoints on first use, under those projects' own licence terms.

Heavy dependencies are imported lazily inside methods so that the geometry
core, the CLI help text and the test suite remain importable on machines with
no GPU, no PyTorch and no network access.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from .exceptions import BackendError
from .stl_io import TriangleArray

__all__ = [
    "GenerationConfig",
    "HunyuanBackend",
    "HunyuanLargeBackend",
    "MeshGenerator",
    "available_backends",
    "get_backend",
    "register_backend",
    "resolve_device",
    "scale_to_millimetres",
]


def resolve_device(requested: str = "auto") -> str:
    """Select a compute device, falling back to CPU when none is available.

    CUDA is never assumed. ``"auto"`` prefers CUDA, then Apple MPS, then CPU.
    An explicitly requested device that is unavailable raises rather than
    silently downgrading, so a user who asked for a GPU is told they did not
    get one.

    Parameters
    ----------
    requested : {'auto', 'cuda', 'mps', 'cpu'}, optional
        Desired device.

    Returns
    -------
    str
        Device string suitable for ``torch``.

    Raises
    ------
    BackendError
        If PyTorch is missing, or a specific device was requested and is
        unavailable.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise BackendError(
            "PyTorch is not installed. Mesh generation requires it; the "
            "geometry and validation tools do not."
        ) from exc

    has_cuda = bool(torch.cuda.is_available())
    has_mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())

    if requested == "auto":
        if has_cuda:
            return "cuda"
        if has_mps:
            return "mps"
        return "cpu"

    if requested == "cuda" and not has_cuda:
        raise BackendError("CUDA was requested but torch.cuda.is_available() is False")
    if requested == "mps" and not has_mps:
        raise BackendError("MPS was requested but is not available")
    if requested not in {"cuda", "mps", "cpu"}:
        raise BackendError(f"unknown device {requested!r}")
    return requested


@dataclass(frozen=True)
class GenerationConfig:
    """Parameters controlling a generation run.

    Attributes
    ----------
    backend : str
        Registered backend name.
    checkpoint : str
        Backend-specific checkpoint identifier, for example ``"default"`` or
        ``"large"``.
    device : str
        ``"auto"``, ``"cuda"``, ``"mps"`` or ``"cpu"``.
    fp16 : bool
        Request half precision. Honoured only on CUDA; ignored elsewhere.
    steps : int
        Denoising steps.
    resolution : int
        Volume grid resolution for surface extraction. Detail smaller than one
        cell cannot be represented.
    guidance : float
        Classifier-free guidance scale.
    seed : int
        Random seed, for reproducibility.
    max_faces : int
        Upper bound on output triangle count.
    remove_background : bool
        Apply automatic background removal to inputs.
    target_size_mm : float
        Longest bounding-box dimension of the exported mesh, in millimetres.
    """

    backend: str = "hunyuan3d"
    checkpoint: str = "default"
    device: str = "auto"
    fp16: bool = False
    steps: int = 30
    resolution: int = 256
    guidance: float = 5.0
    seed: int = 1234
    max_faces: int = 200_000
    remove_background: bool = True
    target_size_mm: float = 30.0

    def cache_key(self, images: Sequence[Path]) -> str:
        """Return a deterministic cache key for these inputs and settings.

        Combines the content hash of every input image with the parameters that
        affect geometry. Parameters that do not affect geometry are excluded so
        that they do not invalidate the cache.
        """
        digest = hashlib.sha256()
        for image in images:
            digest.update(Path(image).read_bytes())
        digest.update(
            json.dumps(
                {
                    "backend": self.backend,
                    "checkpoint": self.checkpoint,
                    "steps": self.steps,
                    "resolution": self.resolution,
                    "guidance": self.guidance,
                    "seed": self.seed,
                    "max_faces": self.max_faces,
                    "remove_background": self.remove_background,
                    "target_size_mm": self.target_size_mm,
                    "n_images": len(images),
                },
                sort_keys=True,
            ).encode()
        )
        return digest.hexdigest()[:16]


def scale_to_millimetres(
    triangles: TriangleArray, target_longest_mm: float
) -> TriangleArray:
    """Scale a mesh so its longest axis measures `target_longest_mm`.

    The result is centred in x and y and rests on ``z = 0``, which is the
    convention expected by most slicers.

    Raises
    ------
    ValueError
        If the mesh has zero extent in every axis.
    """
    flat = triangles.reshape(-1, 3)
    low, high = flat.min(axis=0), flat.max(axis=0)
    extents = high - low
    longest = float(extents.max())
    if longest <= 0.0:
        raise ValueError("mesh has zero extent and cannot be scaled")

    scaled = triangles * (target_longest_mm / longest)
    flat = scaled.reshape(-1, 3)
    low, high = flat.min(axis=0), flat.max(axis=0)
    offset = np.array([-(low[0] + high[0]) / 2.0, -(low[1] + high[1]) / 2.0, -low[2]])
    return scaled + offset


class MeshGenerator(ABC):
    """Interface implemented by every image-to-mesh backend.

    A backend converts one or more images into a triangle array in arbitrary
    units. Physical scaling, validation and symmetry processing happen
    downstream and are identical for all backends.
    """

    #: Registry name used by :func:`get_backend`.
    name: str = "abstract"

    #: Checkpoint identifiers this backend accepts.
    checkpoints: tuple[str, ...] = ("default",)

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        self._pipeline: Any | None = None

    @abstractmethod
    def load(self) -> None:
        """Load model weights, downloading them if required."""

    @abstractmethod
    def generate(self, images: Sequence[Any]) -> TriangleArray:
        """Generate a triangle array from one or more preprocessed images."""

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of this backend instance."""
        return {
            "backend": self.name,
            "checkpoint": self.config.checkpoint,
            "device": self.config.device,
            "fp16": self.config.fp16,
        }


_REGISTRY: dict[str, type[MeshGenerator]] = {}


def register_backend(cls: type[MeshGenerator]) -> type[MeshGenerator]:
    """Register a backend class under its :attr:`MeshGenerator.name`."""
    _REGISTRY[cls.name] = cls
    return cls


def available_backends() -> tuple[str, ...]:
    """Return the names of all registered backends."""
    return tuple(sorted(_REGISTRY))


def get_backend(config: GenerationConfig) -> MeshGenerator:
    """Instantiate the backend named by `config`.

    Raises
    ------
    BackendError
        If the name is not registered or the checkpoint is not supported.
    """
    try:
        cls = _REGISTRY[config.backend]
    except KeyError as exc:
        raise BackendError(
            f"unknown backend {config.backend!r}; available: "
            f"{', '.join(available_backends())}"
        ) from exc

    if config.checkpoint not in cls.checkpoints:
        raise BackendError(
            f"backend {cls.name!r} does not provide checkpoint "
            f"{config.checkpoint!r}; available: {', '.join(cls.checkpoints)}"
        )
    return cls(config)


def _as_triangle_array(mesh: Any) -> TriangleArray:
    """Convert a backend mesh object into a triangle corner array."""
    vertices = getattr(mesh, "vertices", None)
    faces = getattr(mesh, "faces", None)
    if vertices is None or faces is None:
        raise BackendError(
            f"backend returned {type(mesh).__name__}, which exposes no "
            "'vertices'/'faces' attributes"
        )
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    return vertices[faces]


@register_backend
class HunyuanBackend(MeshGenerator):
    """Tencent Hunyuan3D, shape generation only.

    Texture synthesis is CUDA-only and is not attempted; geometry is the sole
    concern of this package. Weights are downloaded on first use under the
    Tencent Hunyuan licence, which is not the licence of this repository.

    The default checkpoint is the compact model, chosen so the pipeline runs on
    consumer hardware including Apple Silicon. See :class:`HunyuanLargeBackend`
    for the higher-capacity variant.
    """

    name = "hunyuan3d"
    checkpoints = ("default",)

    #: Repository and subfolder per checkpoint identifier.
    _CHECKPOINTS: ClassVar[dict[str, tuple[str, str]]] = {
        "default": ("tencent/Hunyuan3D-2mini", "hunyuan3d-dit-v2-mini-turbo"),
    }

    def _repo(self) -> tuple[str, str]:
        return self._CHECKPOINTS[self.config.checkpoint]

    def load(self) -> None:
        """Load the diffusion pipeline onto the resolved device."""
        # Missing MPS kernels fall back to CPU rather than aborting the run.
        # Must be set before torch is imported.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

        try:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline as Pipeline
        except ImportError as exc:
            raise BackendError(
                "hy3dgen is not installed. Run scripts/setup.sh to install the "
                "generation dependencies."
            ) from exc

        device = resolve_device(self.config.device)
        repo, subfolder = self._repo()

        kwargs = _accepted_kwargs(
            Pipeline.from_pretrained,
            {"subfolder": subfolder, "use_safetensors": True, "device": device},
        )
        pipeline = Pipeline.from_pretrained(repo, **kwargs)

        if "device" not in kwargs and hasattr(pipeline, "to"):
            pipeline = pipeline.to(device)

        if self.config.fp16 and device == "cuda" and hasattr(pipeline, "half"):
            pipeline = pipeline.half()

        self._pipeline = pipeline
        self._device = device

    def generate(self, images: Sequence[Any]) -> TriangleArray:
        """Run diffusion and extract a surface.

        Surface extraction defaults to a differentiable marching-cubes
        implementation that requires CUDA. On other devices plain marching
        cubes is requested explicitly, with a retry as a fallback, so the
        failure does not occur after the diffusion run has completed.
        """
        if self._pipeline is None:
            self.load()
        assert self._pipeline is not None

        import torch

        generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        payload = images[0] if len(images) == 1 else list(images)

        base: dict[str, Any] = {
            "image": payload,
            "num_inference_steps": self.config.steps,
            "octree_resolution": self.config.resolution,
            "guidance_scale": self.config.guidance,
            "generator": generator,
        }
        attempts: list[dict[str, Any]] = (
            [base] if self._device == "cuda" else [{**base, "mc_algo": "mc"}, base]
        )

        last: Exception | None = None
        for index, kwargs in enumerate(attempts):
            try:
                mesh = self._pipeline(
                    **_accepted_kwargs(self._pipeline.__call__, kwargs)
                )[0]
                return _as_triangle_array(self._postprocess(mesh))
            except Exception as exc:
                last = exc
                text = f"{type(exc).__name__}: {exc}".lower()
                recoverable = any(
                    token in text
                    for token in ("diso", "dmc", "cuda", "marching", "not implemented")
                )
                if index + 1 < len(attempts) and recoverable:
                    continue
                raise BackendError(f"generation failed: {exc}") from exc

        raise BackendError(f"generation failed: {last}")

    def _postprocess(self, mesh: Any) -> Any:
        """Remove debris and cap triangle count. Failures are non-fatal."""
        try:
            from hy3dgen.shapegen import (
                DegenerateFaceRemover,
                FaceReducer,
                FloaterRemover,
            )
        except ImportError:
            return mesh

        # Cleanup is best-effort: a backend that lacks one of these routines
        # still yields usable geometry, so failures are suppressed.
        for step in (FloaterRemover(), DegenerateFaceRemover()):
            with contextlib.suppress(Exception):
                mesh = step(mesh)
        with contextlib.suppress(Exception):
            mesh = FaceReducer()(mesh, max_facenum=self.config.max_faces)
        return mesh


@register_backend
class HunyuanLargeBackend(HunyuanBackend):
    """Higher-capacity Hunyuan3D checkpoints for CUDA hardware.

    These checkpoints produce finer surface geometry than the compact model at
    a substantially higher memory cost, and are appropriate for discrete NVIDIA
    GPUs. They are not distributed with this package and are not expected to
    run on Apple Silicon.

    Approximate device memory for shape generation, from the upstream project's
    documentation rather than measurement in this repository:

    ==============  ===================
    Checkpoint      Reported VRAM
    ==============  ===================
    ``standard``    approximately 6 GB
    ``large``       approximately 10 GB
    ==============  ===================
    """

    name = "hunyuan3d-large"
    checkpoints = ("standard", "large")

    _CHECKPOINTS: ClassVar[dict[str, tuple[str, str]]] = {
        "standard": ("tencent/Hunyuan3D-2", "hunyuan3d-dit-v2-0"),
        "large": ("tencent/Hunyuan3D-2", "hunyuan3d-dit-v2-0-turbo"),
    }


def _accepted_kwargs(function: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop keyword arguments a callable does not accept.

    Upstream signatures change between releases, and passing one stale keyword
    raises only after the model has loaded. Filtering against the real
    signature converts that into a no-op. If introspection succeeds but removes
    the essential ``image`` argument, the signature is assumed to belong to a
    wrapper and everything is passed through unchanged.
    """
    import inspect

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return kwargs

    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return kwargs

    kept = {k: v for k, v in kwargs.items() if k in signature.parameters}
    if "image" in kwargs and "image" not in kept:
        return kwargs
    return kept


def load_images(paths: Iterable[Path], remove_background: bool) -> list[Any]:
    """Load images and optionally remove their backgrounds.

    Raises
    ------
    BackendError
        If Pillow is unavailable.
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise BackendError("Pillow is required to load input images") from exc

    remover = None
    if remove_background:
        try:
            from hy3dgen.rembg import BackgroundRemover

            remover = BackgroundRemover()
        except ImportError:
            remover = None

    images: list[Any] = []
    for path in paths:
        image = Image.open(path).convert("RGBA")
        if remover is not None:
            image = remover(image.convert("RGB"))
        images.append(image)
    return images
