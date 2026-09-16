#!/usr/bin/env bash
# scripts/check_formatting.sh
#
# Validates code formatting with black and isort (blocking) and
# Google-style docstrings with pydocstyle (warning only).
#
# Usage: bash scripts/check_formatting.sh
# Exit code: 0 = pass, 1 = black or isort failure.

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

echo "=== Formatting check (black + isort + pydocstyle) ==="

# ---------- black ----------
echo "--- black ---"
if "$PYTHON" -m black --check --target-version py312 --line-length 79 src/; then
    echo "✅  black: OK"
    BLACK_OK=true
else
    echo "❌  black: FAILED  (run: python -m black --target-version py312 --line-length 79 src/)"
    BLACK_OK=false
fi

# ---------- isort ----------
echo "--- isort ---"
if "$PYTHON" -m isort --check-only --line-length 79 src/; then
    echo "✅  isort: OK"
    ISORT_OK=true
else
    echo "❌  isort: FAILED  (run: python -m isort --line-length 79 src/)"
    ISORT_OK=false
fi

# ---------- pydocstyle (warning only) ----------
echo "--- pydocstyle ---"
if "$PYTHON" -m pydocstyle \
       --convention=google \
       --add-ignore=D100,D104,D205,D212,D402,D411 \
       src/ 2>&1; then
    echo "✅  pydocstyle: OK"
else
    echo "⚠️   pydocstyle: warnings (not blocking)"
fi

echo "======================================"

if [ "$BLACK_OK" = false ] || [ "$ISORT_OK" = false ]; then
    echo "❌  Formatting check FAILED"
    exit 1
fi

echo "✅  Formatting check PASSED"
