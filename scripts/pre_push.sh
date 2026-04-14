#!/usr/bin/env bash
# scripts/pre_push.sh
#
# Master validation script — runs every required CI gate locally.
# All AI agents and contributors MUST run this before pushing.
#
# Gates (all required by default):
#   1. check_formatting.sh  — black + isort (blocking), pydocstyle (warning)
#   2. run_tests.sh         — pytest unit tests
#   3. run_examples.sh      — arcoex headless image generation
#   4. run_smoke_tests.sh   — arcosim short headless recordings
#   5. generate_videos.sh   — arcosim full-length simulation videos
#
# Usage: bash scripts/pre_push.sh [options]
#
# Options:
#   --no-examples   Skip run_examples.sh (when matplotlib/display unavailable)
#   --no-smoke      Skip run_smoke_tests.sh (when xvfb/ffmpeg unavailable)
#   --no-videos     Skip generate_videos.sh (when xvfb/ffmpeg unavailable)
#
# Exit code: 0 = all active gates pass, 1 = at least one gate failed.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS="$REPO_ROOT/scripts"

RUN_EXAMPLES=true
RUN_SMOKE=true
RUN_VIDEOS=true

for ARG in "$@"; do
    case $ARG in
        --no-examples) RUN_EXAMPLES=false ;;
        --no-smoke)    RUN_SMOKE=false ;;
        --no-videos)   RUN_VIDEOS=false ;;
    esac
done

echo "╔══════════════════════════════════════╗"
echo "║       ARCO pre-push validation        ║"
echo "╚══════════════════════════════════════╝"

FAILED=0

run_gate() {
    local NAME="$1"
    local SCRIPT="$2"
    shift 2
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Gate: $NAME"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if bash "$SCRIPT" "$@"; then
        echo "✅  $NAME — PASSED"
    else
        echo "❌  $NAME — FAILED"
        FAILED=$((FAILED + 1))
    fi
}

run_gate "Formatting (black + isort)" "$SCRIPTS/check_formatting.sh"
run_gate "Unit tests (pytest)"        "$SCRIPTS/run_tests.sh"

if [ "$RUN_EXAMPLES" = true ]; then
    run_gate "Examples (arcoex)"      "$SCRIPTS/run_examples.sh"
fi

if [ "$RUN_SMOKE" = true ]; then
    run_gate "Smoke tests (arcosim)"  "$SCRIPTS/run_smoke_tests.sh"
fi

if [ "$RUN_VIDEOS" = true ]; then
    run_gate "Videos (arcosim)"       "$SCRIPTS/generate_videos.sh"
fi

echo ""
echo "╔══════════════════════════════════════╗"
if [ $FAILED -eq 0 ]; then
    echo "║  ✅  ALL GATES PASSED               ║"
else
    printf "║  ❌  %d GATE(S) FAILED               ║\n" "$FAILED"
fi
echo "╚══════════════════════════════════════╝"

exit $FAILED
