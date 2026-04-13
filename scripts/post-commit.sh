#!/usr/bin/env bash
# scripts/post-commit.sh
#
# Master validation script — runs every required CI gate locally.
# All AI agents and contributors should run this before pushing.
#
# Gates (all required):
#   1. check_formatting.sh  — black + isort (+ pydocstyle warnings)
#   2. run_tests.sh         — pytest unit tests
#   3. run_examples.sh      — arcoex headless image generation
#   4. run_smoke_tests.sh   — arcosim headless recordings
#
# Usage: bash scripts/post-commit.sh
# Exit code: 0 = all gates pass, 1 = at least one gate failed.
#
# NOTE: run_examples.sh and run_smoke_tests.sh require the [tools] extras
# and, for smoke tests, xvfb + ffmpeg.  Skip them with --no-examples and
# --no-smoke respectively when those dependencies are not available.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS="$REPO_ROOT/scripts"

RUN_EXAMPLES=true
RUN_SMOKE=true

for ARG in "$@"; do
    case $ARG in
        --no-examples) RUN_EXAMPLES=false ;;
        --no-smoke)    RUN_SMOKE=false ;;
    esac
done

echo "╔══════════════════════════════════════╗"
echo "║        ARCO post-commit checks        ║"
echo "╚══════════════════════════════════════╝"

FAILED=0

run_gate() {
    local NAME="$1"
    local SCRIPT="$2"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Gate: $NAME"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if bash "$SCRIPT"; then
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

echo ""
echo "╔══════════════════════════════════════╗"
if [ $FAILED -eq 0 ]; then
    echo "║  ✅  ALL GATES PASSED               ║"
else
    echo "║  ❌  $FAILED GATE(S) FAILED           ║"
fi
echo "╚══════════════════════════════════════╝"

exit $FAILED
