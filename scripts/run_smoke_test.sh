#!/usr/bin/env bash
# scripts/run_smoke_test.sh
#
# Runs a short headless recording for a single arcosim scenario to verify
# it executes without errors.
#
# Usage: bash scripts/run_smoke_test.sh <scenario> [options]
#
# Options:
#   --duration <secs>       Recording duration in seconds (default: 3)
#   --out-dir <path>        Output directory for the clip (default: /tmp/arco_smoke)
#   --result-file <path>    Write "pass" or "fail" to this file (optional)
#
# Exit code: 0 = pass, 1 = fail.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SCENARIO=""
DURATION=3
OUT_DIR="/tmp/arco_smoke"
RESULT_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --duration)
            [[ -z "${2:-}" || "${2:-}" == -* ]] && echo "Error: --duration requires a value" && exit 1
            DURATION="$2"; shift 2 ;;
        --out-dir)
            [[ -z "${2:-}" || "${2:-}" == -* ]] && echo "Error: --out-dir requires a value" && exit 1
            OUT_DIR="$2"; shift 2 ;;
        --result-file)
            [[ -z "${2:-}" || "${2:-}" == -* ]] && echo "Error: --result-file requires a value" && exit 1
            RESULT_FILE="$2"; shift 2 ;;
        -*) echo "Unknown option: $1"; exit 1 ;;
        *)  SCENARIO="$1"; shift ;;
    esac
done

if [ -z "$SCENARIO" ]; then
    echo "Usage: $0 <scenario> [--duration <secs>] [--out-dir <dir>] [--result-file <path>]"
    exit 1
fi

MAP_FILE="map/${SCENARIO}.yml"
if [ ! -f "$MAP_FILE" ]; then
    echo "❌  Config not found: $MAP_FILE"
    exit 1
fi

mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/smoke_${SCENARIO}.mp4"

echo "=== Smoke test: $SCENARIO ==="
echo "Map      : $MAP_FILE"
echo "Output   : $OUT"
echo "Duration : ${DURATION} s"

PASSED=false
if SDL_AUDIODRIVER=dummy xvfb-run -a arcosim "$MAP_FILE" \
       -o "$OUT" \
       -d "$DURATION"; then
    PASSED=true
    echo "$SCENARIO: PASSED"
else
    echo "$SCENARIO: FAILED"
fi

if [ -n "$RESULT_FILE" ]; then
    mkdir -p "$(dirname "$RESULT_FILE")"
    if $PASSED; then
        echo "pass" > "$RESULT_FILE"
    else
        echo "fail" > "$RESULT_FILE"
    fi
fi

if $PASSED; then
    exit 0
else
    exit 1
fi
