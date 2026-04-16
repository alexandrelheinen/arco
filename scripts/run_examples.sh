#!/usr/bin/env bash
# scripts/run_examples.sh
#
# Generates static example images for every scenario (headless).
# Uses `arcosim --image --record` (replaces the former `arcoex --save`).
# Requires: package installed with [tools] extras and matplotlib.
#
# Usage: bash scripts/run_examples.sh [--save-dir /path/to/output]
# Exit code: 0 = all examples produced, 1 = at least one failure.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SAVE_DIR="${1:-/tmp/arco_examples}"
mkdir -p "$SAVE_DIR"

echo "=== Example image generation (arcosim --image) ==="
echo "Output directory: $SAVE_DIR"

SCENARIOS=(
    "map/astar.yml"
    "map/ppp.yml"
    "map/rr.yml"
    "map/city.yml"
    "map/vehicle.yml"
    "map/rrp.yml"
    "map/occ.yml"
)

FAILED=0
for CFG in "${SCENARIOS[@]}"; do
    NAME="$(basename "$CFG" .yml)"
    OUT="$SAVE_DIR/arcosim_${NAME}.png"
    echo "--- $NAME ---"
    if MPLBACKEND=Agg arcosim "$CFG" --image --record "$OUT"; then
        echo "✅  $NAME: OK  →  $OUT"
    else
        echo "❌  $NAME: FAILED"
        FAILED=$((FAILED + 1))
    fi
done

echo "======================================"
if [ $FAILED -eq 0 ]; then
    echo "✅  All examples PASSED"
    exit 0
else
    echo "❌  $FAILED example(s) FAILED"
    exit 1
fi

