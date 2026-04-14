#!/usr/bin/env bash
# scripts/run_smoke_tests.sh
#
# Runs a short headless recording of every arcosim simulator to ensure they
# execute without errors.  Requires: xvfb, ffmpeg, libgl1, and the package
# installed with [tools] extras.
#
# Usage: bash scripts/run_smoke_tests.sh [--duration <seconds>]
# Exit code: 0 = all pass, 1 = at least one failure.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DURATION=5
OUT_DIR="/tmp/arco_smoke"
mkdir -p "$OUT_DIR"

echo "=== Simulator smoke tests (arcosim, headless) ==="

SCENARIOS=(
    "src/arco/tools/map/astar.yml"
    "src/arco/tools/map/city.yml"
    "src/arco/tools/map/rr.yml"
    "src/arco/tools/map/vehicle.yml"
    "src/arco/tools/map/ppp.yml"
    "src/arco/tools/map/rrp.yml"
    "src/arco/tools/map/occ.yml"
)

FAILED=0
for CFG in "${SCENARIOS[@]}"; do
    NAME="$(basename "$CFG" .yml)"
    OUT="$OUT_DIR/smoke_${NAME}.mp4"
    echo "--- $NAME ---"
    if SDL_AUDIODRIVER=dummy xvfb-run -a arcosim "$CFG" \
           --fps 30 \
           --record "$OUT" \
           --record-duration "$DURATION"; then
        echo "✅  $NAME: OK"
    else
        echo "❌  $NAME: FAILED"
        FAILED=$((FAILED + 1))
    fi
done

echo "======================================"
if [ $FAILED -eq 0 ]; then
    echo "✅  All smoke tests PASSED"
    exit 0
else
    echo "❌  $FAILED smoke test(s) FAILED"
    exit 1
fi

