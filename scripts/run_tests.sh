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

echo "=== Unit tests (pytest) ==="
python -m pytest tests/ -v --tb=short
EXIT=$?

if [ $EXIT -eq 0 ]; then
    echo "✅  Unit tests PASSED"
else
    echo "❌  Unit tests FAILED"
fi

exit $EXIT
