#!/usr/bin/env bash
# scripts/pre_report_progress.sh
#
# Fast validation gate — runs before every `report_progress` call.
# Must pass before any code is pushed to the PR.
#
# On failure the full output is printed to stdout so that CI can capture
# it and post it as a PR comment (or fail the gate with context).
#
# Gates (all required, fast):
#   1. check_formatting.sh  — black + isort (blocking), pydocstyle (warning)
#
# Usage: bash scripts/pre_report_progress.sh
# Exit code: 0 = all gates pass, 1 = at least one gate failed.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS="$REPO_ROOT/scripts"

echo "╔══════════════════════════════════════════╗"
echo "║    ARCO pre-report-progress validation    ║"
echo "╚══════════════════════════════════════════╝"

FAILED=0
OUTPUT_LOG="$(mktemp)"

run_gate() {
    local NAME="$1"
    local SCRIPT="$2"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Gate: $NAME"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    local GATE_LOG
    GATE_LOG="$(mktemp)"
    if bash "$SCRIPT" 2>&1 | tee "$GATE_LOG"; then
        echo "✅  $NAME — PASSED"
    else
        echo "❌  $NAME — FAILED"
        FAILED=$((FAILED + 1))
        cat "$GATE_LOG" >> "$OUTPUT_LOG"
    fi
    rm -f "$GATE_LOG"
}

run_gate "Formatting (black + isort)" "$SCRIPTS/check_formatting.sh"

echo ""
echo "╔══════════════════════════════════════════╗"
if [ $FAILED -eq 0 ]; then
    echo "║  ✅  ALL GATES PASSED                   ║"
    echo "╚══════════════════════════════════════════╝"
    rm -f "$OUTPUT_LOG"
    exit 0
else
    printf "║  ❌  %d GATE(S) FAILED — push BLOCKED    ║\n" "$FAILED"
    echo "╚══════════════════════════════════════════╝"
    echo ""
    echo "══════════════ FAILURE DETAILS ════════════"
    cat "$OUTPUT_LOG"
    echo "═══════════════════════════════════════════"
    rm -f "$OUTPUT_LOG"
    exit 1
fi
