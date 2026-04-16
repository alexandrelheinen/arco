#!/usr/bin/env bash
# scripts/generate_videos.sh
#
# Generates full-length simulation videos for every arcosim scenario.
# Intended for release packaging; also included as an optional gate in
# pre_push.sh.
#
# Requires: xvfb, ffmpeg, libgl1, and the package installed with [tools]
# extras.
#
# Usage: bash scripts/generate_videos.sh [options]
#
# Options:
#   --out-dir <path>        Output directory (default: /tmp/arco_videos)
#   --duration <seconds>    Recording duration per scenario (default: 60)
#   --only <name,...>       Comma-separated list of scenario names to run
#                           (default: all). E.g. --only ppp,rr
#
# Exit code: 0 = all pass, 1 = at least one failure.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="/tmp/arco_videos"
DURATION=60
ONLY=""

# Parse optional args
while [[ $# -gt 0 ]]; do
    case $1 in
        --out-dir)  OUT_DIR="$2"; shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        --only)     ONLY="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUT_DIR"

echo "=== Simulation video generation (arcosim, headless) ==="
echo "Output directory : $OUT_DIR"
echo "Duration per clip: ${DURATION}s"
[ -n "$ONLY" ] && echo "Scenarios filter : $ONLY"

ALL_SCENARIOS=(
    "map/astar.yml"
    "map/city.yml"
    "map/rr.yml"
    "map/vehicle.yml"
    "map/ppp.yml"
    "map/rrp.yml"
    "map/occ.yml"
)

FAILED=0
for CFG in "${ALL_SCENARIOS[@]}"; do
    NAME="$(basename "$CFG" .yml)"

    # Apply --only filter if provided
    if [ -n "$ONLY" ]; then
        IFS=',' read -ra FILTER <<< "$ONLY"
        MATCH=false
        for F in "${FILTER[@]}"; do
            [ "$F" = "$NAME" ] && MATCH=true && break
        done
        [ "$MATCH" = false ] && continue
    fi

    OUT="$OUT_DIR/arcosim_${NAME}.mp4"
    echo "--- $NAME ---"
    if SDL_AUDIODRIVER=dummy xvfb-run -a arcosim "$CFG" \
           --fps 30 \
           --record "$OUT" \
           --record-duration "$DURATION"; then
        echo "✅  $NAME: OK  →  $OUT"
    else
        echo "❌  $NAME: FAILED"
        FAILED=$((FAILED + 1))
    fi
done

echo "======================================"
if [ $FAILED -eq 0 ]; then
    echo "✅  All videos GENERATED"
    exit 0
else
    echo "❌  $FAILED video(s) FAILED"
    exit 1
fi

