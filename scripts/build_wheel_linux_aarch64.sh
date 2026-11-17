#!/usr/bin/env bash
# scripts/build_wheel_linux_aarch64.sh
#
# Cross-build one linux aarch64 abi3 manylinux2014 wheel on an x86_64
# host with maturin --zig (FR-BUILD-07).  See docs/build-aarch64.md.
#
# Usage: bash scripts/build_wheel_linux_aarch64.sh [source-dir]
#   source-dir  tree to build (default: the repo that holds this script).
#               Pass a worktree of tag v0.5.0 to attach that release.
#
# Exit code: 0 = one correctly tagged wheel in <source-dir>/dist/,
#            127 = missing prerequisite, non-zero = build failure.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCE_DIR="$(cd "${1:-$SCRIPT_ROOT}" && pwd)"
cd "$SOURCE_DIR"

have() { command -v "$1" >/dev/null 2>&1; }

require() {
    if ! have "$1"; then
        echo "missing required tool: $1" >&2
        echo "install it with: $2" >&2
        exit 127
    fi
}

require rustup "https://rustup.rs"
require cargo "https://rustup.rs"

PYTHON="${ARCO_PYTHON:-}"
if [ -z "$PYTHON" ]; then
    if [ -x "$SCRIPT_ROOT/.venv/bin/python" ]; then
        PYTHON="$SCRIPT_ROOT/.venv/bin/python"
    elif [ -x "$SOURCE_DIR/.venv/bin/python" ]; then
        PYTHON="$SOURCE_DIR/.venv/bin/python"
    else
        PYTHON="$(command -v python3)"
    fi
fi

if ! "$PYTHON" -c "import maturin" >/dev/null 2>&1 && ! have maturin; then
    echo "missing required tool: maturin" >&2
    echo "install it with: $PYTHON -m pip install 'maturin>=1.7,<2'" >&2
    exit 127
fi

if ! have zig; then
    # ziglang from PyPI ships the binary next to the package, not on PATH.
    ZIG_FROM_PIP="$("$PYTHON" - <<'PY'
import pathlib
import sys
try:
    import ziglang
except ImportError:
    sys.exit(1)
print(pathlib.Path(ziglang.__file__).resolve().parent)
PY
)" || true
    if [ -n "${ZIG_FROM_PIP:-}" ] && [ -x "$ZIG_FROM_PIP/zig" ]; then
        export PATH="$ZIG_FROM_PIP:$PATH"
    else
        echo "missing required tool: zig (ziglang)" >&2
        echo "install it with: $PYTHON -m pip install ziglang" >&2
        exit 127
    fi
fi

export PATH="$HOME/.cargo/bin:$PATH"

if ! rustup target list --installed | grep -qx 'aarch64-unknown-linux-gnu'; then
    echo "rustup target aarch64-unknown-linux-gnu is not installed" >&2
    echo "install it with: rustup target add aarch64-unknown-linux-gnu" >&2
    exit 127
fi

mkdir -p dist
# Drop prior aarch64 wheels so the "exactly one" check is meaningful.
shopt -s nullglob
for old in dist/arco-*-manylinux*_aarch64*.whl; do
    rm -f "$old"
done
shopt -u nullglob

echo "=== building linux aarch64 wheel from $SOURCE_DIR ==="
START="$(date +%s)"

MATURIN_CMD=("$PYTHON" -m maturin)
if ! "$PYTHON" -c "import maturin" >/dev/null 2>&1; then
    MATURIN_CMD=(maturin)
fi

"${MATURIN_CMD[@]}" build \
    --release \
    --target aarch64-unknown-linux-gnu \
    --zig \
    --compatibility manylinux2014 \
    --out dist

END="$(date +%s)"
echo "=== build wall time: $((END - START))s ==="

shopt -s nullglob
wheels=(dist/arco-*-cp310-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl)
shopt -u nullglob

if [ "${#wheels[@]}" -ne 1 ]; then
    echo "expected exactly one aarch64 abi3 manylinux2014 wheel in dist/" >&2
    echo "found ${#wheels[@]}: ${wheels[*]:-}" >&2
    exit 1
fi

# Prefer the checker from the script tree (main) so a v0.5.0 worktree
# without the new tools still gets the tag assert.
CHECKER="$SCRIPT_ROOT/tools/wheel_check.py"
if [ ! -f "$CHECKER" ]; then
    CHECKER="$SOURCE_DIR/tools/wheel_check.py"
fi
"$PYTHON" "$CHECKER" "${wheels[0]}"

echo "built ${wheels[0]}"
ls -l "${wheels[0]}"
sha256sum "${wheels[0]}"
