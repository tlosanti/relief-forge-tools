#!/bin/bash
# Install optional mesh-generation dependencies.
#
# The geometry core requires only NumPy and is already usable without this
# script. Everything installed here is needed solely for image-to-mesh
# inference.
#
# Two modes, selected automatically:
#
#   LINKED     A sibling Half-Tone-Depth-Wrap virtualenv is present and has
#              PyTorch. Its packages are reused through a .pth file rather than
#              reinstalled. That environment is never modified.
#
#   STANDALONE No sibling environment. A self-contained virtualenv is created
#              and PyTorch installed into it.
#
#   ./scripts/setup.sh
#
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
cd "$ROOT"

SHARED="$(dirname "$ROOT")/Half-Tone-Depth-Wrap/.venv"
VENV="$ROOT/.venv-gen3d"
SRC="$ROOT/.gen3d-src"

echo "==> relief-forge-tools: generation dependencies"
echo "    root: $ROOT"

MODE="standalone"
if [ -x "$SHARED/bin/python" ] && "$SHARED/bin/python" -c 'import torch' 2>/dev/null; then
  MODE="linked"
fi

if [ "$MODE" = "linked" ]; then
  SHARED_SITE="$("$SHARED/bin/python" -c 'import site;print(site.getsitepackages()[0])')"
  BASE_PY="$SHARED/bin/python"
  PYV="$("$BASE_PY" -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
  echo "    mode: linked (python $PYV, torch $("$BASE_PY" -c 'import torch;print(torch.__version__)'))"
else
  BASE_PY="$(command -v python3)"
  PYV="$("$BASE_PY" -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
  echo "    mode: standalone (python $PYV)"
fi

# The virtualenv is built from the interpreter that owns the packages being
# linked. Compiled extensions such as PyTorch are tied to a specific Python
# minor version, and a mismatch would surface later as an obscure ImportError
# rather than failing here.
if [ ! -d "$VENV" ]; then
  echo "==> creating .venv-gen3d (python $PYV)"
  "$BASE_PY" -m venv "$VENV"
fi

LOCAL_PYV="$("$VENV/bin/python" -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
if [ "$LOCAL_PYV" != "$PYV" ]; then
  echo "!! .venv-gen3d is python $LOCAL_PYV but $PYV was expected."
  echo "   rm -rf '$VENV' && ./scripts/setup.sh"
  exit 1
fi

"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel

if [ "$MODE" = "linked" ]; then
  # A .pth is used rather than `venv --system-site-packages`: a virtualenv
  # created from another virtualenv inherits the base interpreter's packages,
  # not the parent's. Locally installed packages still take precedence.
  LOCAL_SITE="$("$VENV/bin/python" -c 'import site;print(site.getsitepackages()[0])')"
  echo "$SHARED_SITE" > "$LOCAL_SITE/_shared_venv.pth"
  if ! "$VENV/bin/python" -c 'import torch' 2>/dev/null; then
    echo "!! torch not reachable through the .pth link ($SHARED_SITE)"
    exit 1
  fi
  echo "    torch reachable from .venv-gen3d"
else
  echo "==> installing torch"
  # CUDA builds are selected by index URL; the default wheel covers CPU and
  # Apple Silicon. See the Hardware Acceleration section of the README.
  "$VENV/bin/pip" install torch torchvision
fi

echo "==> installing required packages"
"$VENV/bin/pip" install -e ".[generation]"

echo "==> installing optional packages (failures here are survivable)"
for pkg in "rembg[cpu]" onnxruntime pymeshlab; do
  echo "    -- $pkg"
  if ! "$VENV/bin/pip" install "$pkg"; then
    case "$pkg" in
      "rembg[cpu]"|onnxruntime)
        echo "    [warn] $pkg unavailable; run generate.py with --no-rembg" ;;
      pymeshlab)
        echo "    [warn] $pkg unavailable; backend cleanup routines are used instead" ;;
    esac
  fi
done

if [ ! -d "$SRC" ]; then
  echo "==> cloning Hunyuan3D-2"
  git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git "$SRC"
fi

echo "==> installing hy3dgen (shape generation only)"
cd "$SRC"
"$VENV/bin/pip" install -e . || {
  echo "!! hy3dgen install failed. Texture and rasterizer extensions are"
  echo "   CUDA-only and are expected to fail on Apple Silicon; shape"
  echo "   generation does not require them."
  exit 1
}

cd "$ROOT"
echo
echo "==> verify:"
echo "     .venv-gen3d/bin/python cli/generate.py --selftest"
