#!/usr/bin/env bash
# scripts/generate_diagrams.sh
#
# Generates pyreverse class and package diagrams for the arco source tree.
# Requires: pyreverse (included with pylint) and graphviz.
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

echo "=== Generating pyreverse diagrams ==="
echo "Output directory: $OUT_DIR"

# --ignore=pipeline.py avoids a pyreverse KeyError triggered by modules
# that are re-exported from a parent __init__.py but also appear as a
# direct "from .pipeline import …" dependency inside that __init__.
echo "--- Class diagram ---"
pyreverse --verbose -f PUB_ONLY --colorized -o png -k --no-standalone \
    --ignore=pipeline.py \
    -p classes ./src/arco/
mv classes_classes.png "$OUT_DIR/pyreverse_classes.png"
echo "✅  pyreverse_classes.png  →  $OUT_DIR"

echo "--- Package diagram ---"
pyreverse --verbose -f PUB_ONLY --colorized -o png -k --no-standalone \
    --ignore=pipeline.py \
    -p packages ./src/arco/
mv packages_packages.png "$OUT_DIR/pyreverse_packages.png"
echo "✅  pyreverse_packages.png  →  $OUT_DIR"

echo "======================================"
echo "✅  All diagrams GENERATED"
