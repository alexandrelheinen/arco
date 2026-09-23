#!/usr/bin/env bash
# scripts/generate_release_images.sh
#
# Renders the quiet web image set (docs/proposals/web-illustrations.md).
# Intended for the release workflow. Requires the package installed with
# the dev extra (matplotlib) and a built arco extension.
#
# Usage: bash scripts/generate_release_images.sh [options]
#
# Options:
#   --out-dir <path>   Output directory (default: /tmp/arco_release_images)
#   --dry-run          Print the render command and the filename list, then exit
#
# Exit code: 0 = rendered (or listed), non-zero = renderer failed.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="/tmp/arco_release_images"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

PYTHON="${PYTHON:-python3}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

echo "=== Release image generation ==="
echo "Output directory : $OUT_DIR"
echo "Command          : $PYTHON tools/render_web.py --output $OUT_DIR"

if [ "$DRY_RUN" -eq 1 ]; then
    echo "Filenames:"
    "$PYTHON" tools/render_web.py --list
    exit 0
fi

mkdir -p "$OUT_DIR"
"$PYTHON" tools/render_web.py --output "$OUT_DIR"
echo "✅  Release images written to $OUT_DIR"
