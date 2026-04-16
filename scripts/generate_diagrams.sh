#!/usr/bin/env bash
# scripts/generate_diagrams.sh
#
# Generates graphviz architecture diagrams for the arco source tree:
#
#   1. overview.png       — high-level package dependency map (always generated)
#   2. scoped_<pkg>.png   — per-module class diagrams for files changed vs main
#
# Requires: graphviz CLI (dot) + the graphviz Python package.
# Install the CLI on Ubuntu with: sudo apt-get install -y graphviz
# Install the Python wrapper with: pip install graphviz
#
# Usage: bash scripts/generate_diagrams.sh [--out-dir <path>]
#
# Exit code: 0 = success, 1 = failure.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="/tmp/arco_diagrams"

while [[ $# -gt 0 ]]; do
    case $1 in
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUT_DIR"

echo "=== Generating graphviz diagrams ==="
echo "Output directory: $OUT_DIR"

# -----------------------------------------------------------------------
# 1. High-level package dependency map (always generated).
# -----------------------------------------------------------------------
echo "--- Package overview ---"
python scripts/render_package_map.py --output "$OUT_DIR/overview.png"

# -----------------------------------------------------------------------
# 2. Scoped per-module class diagrams for files changed vs origin/main.
# -----------------------------------------------------------------------
echo "--- Scoped class diagrams ---"

# Detect changed Python files.  Falls back gracefully when origin/main is
# unavailable (e.g. workflow_dispatch on a shallow clone).
CHANGED=""
if git rev-parse --verify origin/main >/dev/null 2>&1; then
    CHANGED=$(git diff --name-only --diff-filter=d origin/main...HEAD | grep '\.py$' || true)
elif git rev-parse --verify origin/HEAD >/dev/null 2>&1; then
    CHANGED=$(git diff --name-only --diff-filter=d origin/HEAD...HEAD | grep '\.py$' || true)
fi

if [ -n "$CHANGED" ]; then
    echo "Changed Python files:"
    echo "$CHANGED"
    # shellcheck disable=SC2086
    echo "$CHANGED" | tr '\n' '\0' | xargs -0 \
        python scripts/render_scoped.py --output-dir "$OUT_DIR"
else
    echo "No changed Python files detected — skipping scoped diagrams."
fi

echo "======================================"
echo "✅  All diagrams GENERATED"
