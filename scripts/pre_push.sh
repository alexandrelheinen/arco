#!/usr/bin/env bash
# scripts/pre_push.sh
#
# Master validation script — runs every required CI gate locally.
# All AI agents and contributors MUST run this before pushing.
#
# Gates (all required — none may be skipped):
#   1. check_formatting.sh  — black + isort (blocking), pydocstyle (warning)
#   2. run_tests.sh         — pytest unit tests
#   3. run_examples.sh      — arcoex headless image generation
#   4. run_smoke_tests.sh   — arcosim short headless recordings
#   5. generate_videos.sh   — arcosim full-length simulation videos
#
# Usage: bash scripts/pre_push.sh
# Exit code: 0 = all gates pass, 1 = at least one gate failed.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS="$REPO_ROOT/scripts"

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
run_gate "Examples (arcoex)"          "$SCRIPTS/run_examples.sh"
run_gate "Smoke tests (arcosim)"      "$SCRIPTS/run_smoke_tests.sh"
run_gate "Videos (arcosim)"           "$SCRIPTS/generate_videos.sh"

echo ""
echo "╔══════════════════════════════════════╗"
if [ $FAILED -eq 0 ]; then
    echo "║  ✅  ALL GATES PASSED               ║"
else
    printf "║  ❌  %d GATE(S) FAILED               ║\n" "$FAILED"
fi
echo "╚══════════════════════════════════════╝"

exit $FAILED
