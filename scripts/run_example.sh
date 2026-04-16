#!/usr/bin/env bash
# scripts/run_example.sh
#
# Generates a static example image for a single arcoex scenario (headless).
# Requires: package installed with [tools] extras and matplotlib.
#
# Usage: bash scripts/run_example.sh <scenario> [--save-dir <path>]
#
# Exit code: 0 = success, 1 = failure.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SCENARIO=""
SAVE_DIR="/tmp/arco_examples"

while [[ $# -gt 0 ]]; do
    case $1 in
        --save-dir)
            [[ -z "${2:-}" || "${2:-}" == -* ]] && echo "Error: --save-dir requires a value" && exit 1
            SAVE_DIR="$2"; shift 2 ;;
        -*) echo "Unknown option: $1"; exit 1 ;;
        *)  SCENARIO="$1"; shift ;;
    esac
done

if [ -z "$SCENARIO" ]; then
    echo "Usage: $0 <scenario> [--save-dir <path>]"
    exit 1
fi

CFG="src/arco/tools/map/${SCENARIO}.yml"
if [ ! -f "$CFG" ]; then
    echo "❌  Config not found: $CFG"
    exit 1
fi

mkdir -p "$SAVE_DIR"
OUT="$SAVE_DIR/arcoex_${SCENARIO}.png"

echo "=== Example image: $SCENARIO ==="
echo "Config : $CFG"
echo "Output : $OUT"

MPLBACKEND=Agg arcoex "$CFG" --save "$OUT"
echo "✅  $SCENARIO: OK  →  $OUT"
