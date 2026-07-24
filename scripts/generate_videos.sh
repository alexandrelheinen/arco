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
#                           (default: all primary scenarios). E.g. --only ppp,city
#   --release               Release mode: use reduced city planner budgets
#                           (map/city_mpc_preview.yml → arcosim_city.mp4) and
#                           pass --fast-record to skip tree-reveal pacing.
#   --dry-run               Print the resolved map / output / flags and exit
#                           without running arcosim (for CI / script tests).
#
# Exit code: 0 = all pass, 1 = at least one failure.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="/tmp/arco_videos"
DURATION=120  # seconds (2 minutes)
ONLY=""
RELEASE=0
DRY_RUN=0

# Primary release / smoke scenario names (basename of map/*.yml).
# Preview / alternate maps are not listed here; --release remaps city.
PRIMARY_SCENARIOS=(city ppp rrp occ)

# Parse optional args
while [[ $# -gt 0 ]]; do
    case $1 in
        --out-dir)  OUT_DIR="$2"; shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        --only)     ONLY="$2"; shift 2 ;;
        --release)  RELEASE=1; shift ;;
        --dry-run)  DRY_RUN=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUT_DIR"

echo "=== Simulation video generation (arcosim, headless) ==="
echo "Output directory : $OUT_DIR"
echo "Duration per clip: ${DURATION} s"
[ "$RELEASE" -eq 1 ] && echo "Mode             : release (city preview map + fast-record)"
[ -n "$ONLY" ] && echo "Scenarios filter : $ONLY"

# Resolve which logical scenario names to run.
SCENARIOS=()
if [ -n "$ONLY" ]; then
    IFS=',' read -ra SCENARIOS <<< "$ONLY"
else
    SCENARIOS=("${PRIMARY_SCENARIOS[@]}")
fi

FAILED=0
for NAME in "${SCENARIOS[@]}"; do
    NAME="$(echo "$NAME" | tr -d '[:space:]')"
    [ -z "$NAME" ] && continue

    # Release city uses the reduced-budget preview map but keeps the public
    # artifact name arcosim_city.mp4 expected by publish_release_videos.sh.
    if [ "$RELEASE" -eq 1 ] && [ "$NAME" = "city" ]; then
        CFG="$REPO_ROOT/map/city_mpc_preview.yml"
    else
        CFG="$REPO_ROOT/map/${NAME}.yml"
    fi

    if [ ! -f "$CFG" ]; then
        echo "❌  $NAME: map not found → $CFG"
        FAILED=$((FAILED + 1))
        continue
    fi

    OUT="$OUT_DIR/arcosim_${NAME}.mp4"
    EXTRA_ARGS=()
    if [ "$RELEASE" -eq 1 ]; then
        EXTRA_ARGS+=(--fast-record)
    fi

    echo "--- $NAME ($CFG) ---"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "DRY-RUN: arcosim $CFG -o $OUT -d $DURATION ${EXTRA_ARGS[*]:-}"
        continue
    fi

    if SDL_AUDIODRIVER=dummy xvfb-run -a arcosim "$CFG" \
           -o "$OUT" \
           -d "$DURATION" \
           "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; then
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
