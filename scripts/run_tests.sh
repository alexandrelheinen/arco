#!/usr/bin/env bash
# scripts/run_tests.sh
#
# Runs the full pytest unit-test suite.
#
# Usage: bash scripts/run_tests.sh
# Exit code: mirrors pytest exit code (0 = all pass).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# The interpreter to run the tools with.  ARCO_PYTHON lets scripts/validate.sh
# hand down the one it already resolved, so a contributor without a bare
# `python` on PATH runs the same checks CI does.
PYTHON="${ARCO_PYTHON:-}"
if [ -z "$PYTHON" ]; then
    if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
        PYTHON="$REPO_ROOT/.venv/bin/python"
    elif command -v python >/dev/null 2>&1; then
        PYTHON=python
    else
        PYTHON=python3
    fi
fi

echo "=== Unit tests (pytest) ==="
"$PYTHON" -m pytest tests/ -v --tb=short
EXIT=$?

if [ $EXIT -eq 0 ]; then
    echo "✅  Unit tests PASSED"
else
    echo "❌  Unit tests FAILED"
fi

exit $EXIT
