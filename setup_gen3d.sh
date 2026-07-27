#!/bin/bash
# Compatibility wrapper. The installer moved to scripts/setup.sh.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
echo "note: setup_gen3d.sh has moved to scripts/setup.sh; forwarding."
exec "$HERE/scripts/setup.sh" "$@"
