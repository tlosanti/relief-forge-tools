#!/bin/bash
# setup_gen3d.sh — install Hunyuan3D shape generation.
#
# Two modes, chosen automatically:
#
#   LINKED     — if the Half-Tone-Depth-Wrap venv is next door, torch and
#                friends (several GB) are reused from it via a .pth file.
#                Nothing is written to that venv; it cannot be broken here.
#
#   STANDALONE — otherwise a self-contained venv is built and torch is
#                installed into it. Slower and larger, but needs nothing else
#                on the machine.
#
#   ./setup_gen3d.sh
#
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# Repo root is this directory when checked out standalone, or its parent when
# living inside relief-forge/tools.
if [ "$(basename "$HERE")" = "tools" ]; then ROOT="$(dirname "$HERE")"; else ROOT="$HERE"; fi
cd "$ROOT"

SHARED="$(dirname "$ROOT")/Half-Tone-Depth-Wrap/.venv"
VENV="$ROOT/.venv-gen3d"
SRC="$ROOT/.gen3d-src"

echo "==> Hunyuan3D setup for Relief Forge"
echo "    root: $ROOT"

# ------------------------------------------------------------------ mode
MODE="standalone"
if [ -x "$SHARED/bin/python" ] && "$SHARED/bin/python" -c 'import torch' 2>/dev/null; then
  MODE="linked"
fi

if [ "$MODE" = "linked" ]; then
  SHARED_SITE="$("$SHARED/bin/python" -c 'import site;print(site.getsitepackages()[0])')"
  BASE_PY="$SHARED/bin/python"
  PYV="$("$BASE_PY" -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
  TORCH_V="$("$BASE_PY" -c 'import torch;print(torch.__version__)')"
  echo "    mode: LINKED — reusing the halftone app's packages"
  echo "    python $PYV, torch $TORCH_V (inherited, not reinstalled)"
else
  BASE_PY="$(command -v python3)"
  PYV="$("$BASE_PY" -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
  echo "    mode: STANDALONE — no sibling venv found, installing torch here"
  echo "    python $PYV at $BASE_PY"
  if [ "$PYV" = "3.14" ]; then
    echo "    [note] python 3.14 is very new; if a package fails to build,"
    echo "           installing python 3.12 and re-running is the quick fix."
  fi
fi

# ------------------------------------------------------------------ venv
# Built from the SAME interpreter that owns the packages being linked.
# Compiled extensions like torch are tied to a specific Python minor version,
# so a mismatch would not fail here -- it would fail later with a confusing
# ImportError.
if [ ! -d "$VENV" ]; then
  echo "==> Creating .venv-gen3d (python $PYV)"
  "$BASE_PY" -m venv "$VENV"
fi

LOCAL_PYV="$("$VENV/bin/python" -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
if [ "$LOCAL_PYV" != "$PYV" ]; then
  echo "!! Version mismatch: .venv-gen3d is python $LOCAL_PYV, expected $PYV."
  echo "   An older .venv-gen3d is in the way. Remove and re-run:"
  echo "     rm -rf '$VENV' && ./setup_gen3d.sh"
  exit 1
fi

"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel

if [ "$MODE" = "linked" ]; then
  # A .pth is used rather than `venv --system-site-packages` because a venv
  # created from another venv inherits the *base* interpreter's packages, not
  # the parent venv's. Local packages still win: .pth entries are appended
  # after the local directory.
  LOCAL_SITE="$("$VENV/bin/python" -c 'import site;print(site.getsitepackages()[0])')"
  echo "$SHARED_SITE" > "$LOCAL_SITE/_shared_venv.pth"

  if ! "$VENV/bin/python" -c 'import torch' 2>/dev/null; then
    echo "!! torch is not visible through the .pth link. Expected it at:"
    echo "     $SHARED_SITE"
    exit 1
  fi
  echo "    torch reachable from .venv-gen3d — link works"
else
  echo "==> Installing torch (this is the big one)"
  "$VENV/bin/pip" install torch torchvision
fi

# ------------------------------------------------------------------ deps
# Only the first group is required. The rest are installed one at a time and
# allowed to fail, since prebuilt wheels lag new Python releases.
echo "==> Installing required packages"
"$VENV/bin/pip" install trimesh numpy pillow einops omegaconf transformers

echo "==> Installing optional packages (failures here are survivable)"
for pkg in "rembg[cpu]" onnxruntime pymeshlab; do
  echo "    -- $pkg"
  if ! "$VENV/bin/pip" install "$pkg"; then
    echo "    [warn] $pkg unavailable on python $PYV — continuing."
    case "$pkg" in
      "rembg[cpu]"|onnxruntime)
        echo "           Background removal is off; run gen3d.py with --no-rembg"
        echo "           and hand it an already cut-out image." ;;
      pymeshlab)
        echo "           Mesh cleanup falls back to hy3dgen's own routines." ;;
    esac
  fi
done

# ---------------------------------------------------------------- hy3dgen
if [ ! -d "$SRC" ]; then
  echo "==> Cloning Hunyuan3D-2"
  git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git "$SRC"
fi

echo "==> Installing hy3dgen (shape only — texture needs CUDA, skipped)"
cd "$SRC"
"$VENV/bin/pip" install -e . || {
  echo "!! hy3dgen install failed."
  echo "   The texture/rasterizer extensions are CUDA-only and are expected to"
  echo "   fail on Apple Silicon; shape generation does not need them. If the"
  echo "   failure is in hy3dpaint or custom_rasterizer, it is safe to ignore."
  exit 1
}

cd "$ROOT"
GEN3D="tools/gen3d.py"
[ -f "$GEN3D" ] || GEN3D="gen3d.py"

echo
echo "==> Done. Verify:"
echo "     .venv-gen3d/bin/python $GEN3D --selftest"
echo
echo "    Then run a photo through it:"
echo "     .venv-gen3d/bin/python $GEN3D photo.jpg --size 30"
