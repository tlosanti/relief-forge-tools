#!/bin/bash
# compare_latest.sh — mesh_check the two most recent STL exports against each other.
#
# Saves fishing timestamped paths with spaces out of the exports folder by hand.
# Older mesh is A, newer is B, so the "B vs A" column reads as "what changed".
#
#   ./tools/compare_latest.sh
#   ./tools/compare_latest.sh ~/some/other/folder
#
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
if [ "$(basename "$HERE")" = "tools" ]; then ROOT="$(dirname "$HERE")"; else ROOT="$HERE"; fi

EXPORTS="${1:-$HOME/Desktop/Filter Exports}"

if [ ! -d "$EXPORTS" ]; then
  echo "No exports folder at: $EXPORTS" >&2
  exit 2
fi

# Null-delimited so spaces in the timestamped folder names cannot split paths.
FILES=()
while IFS= read -r -d '' f; do FILES+=("$f"); done < <(
  find "$EXPORTS" -name '*.stl' -type f -print0 2>/dev/null |
    xargs -0 ls -t 2>/dev/null |
    head -2 |
    tr '\n' '\0'
)

if [ "${#FILES[@]}" -lt 2 ]; then
  echo "Need two STL exports to compare; found ${#FILES[@]} in:" >&2
  echo "  $EXPORTS" >&2
  exit 2
fi

NEW="${FILES[0]}"
OLD="${FILES[1]}"

echo "A (older): $OLD"
echo "B (newer): $NEW"
echo

exec python3 "$HERE/mesh_check.py" "$OLD" --compare "$NEW"
