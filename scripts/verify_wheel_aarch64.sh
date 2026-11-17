#!/usr/bin/env bash
# scripts/verify_wheel_aarch64.sh
#
# Tag-check a linux aarch64 wheel, then install it with
# --only-binary=:all: inside fresh linux/arm64 containers for CPython
# 3.10 and 3.12 and run scripts/wheel_smoke.py (FR-BUILD-06/08).
#
# Usage:
#   bash scripts/verify_wheel_aarch64.sh <wheel> [--expect-version X.Y.Z]
#
# Prerequisites: Docker with arm64 binfmt registered, e.g.
#   docker run --privileged --rm tonistiigi/binfmt --install arm64
#
# Exit code: 0 = both interpreters passed; non-zero = failure.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

WHEEL=""
EXPECT_VERSION=""

usage() {
    echo "Usage: bash scripts/verify_wheel_aarch64.sh <wheel> [--expect-version X.Y.Z]" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --expect-version)
            [[ -z "${2:-}" || "${2:-}" == -* ]] && {
                echo "Error: --expect-version requires a value" >&2
                exit 1
            }
            EXPECT_VERSION="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
        *)
            if [ -n "$WHEEL" ]; then
                echo "Unexpected argument: $1" >&2
                usage
                exit 1
            fi
            WHEEL="$1"
            shift
            ;;
    esac
done

if [ -z "$WHEEL" ]; then
    usage
    exit 1
fi

WHEEL="$(cd "$(dirname "$WHEEL")" && pwd)/$(basename "$WHEEL")"
if [ ! -f "$WHEEL" ]; then
    echo "wheel not found: $WHEEL" >&2
    exit 1
fi

have() { command -v "$1" >/dev/null 2>&1; }

require() {
    if ! have "$1"; then
        echo "missing required tool: $1" >&2
        echo "install it with: $2" >&2
        exit 127
    fi
}

require docker "https://docs.docker.com/engine/install/"

PYTHON="${ARCO_PYTHON:-}"
if [ -z "$PYTHON" ]; then
    if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
        PYTHON="$REPO_ROOT/.venv/bin/python"
    else
        PYTHON="$(command -v python3)"
    fi
fi

CHECK_ARGS=("$WHEEL")
if [ -n "$EXPECT_VERSION" ]; then
    CHECK_ARGS+=(--expect-version "$EXPECT_VERSION")
fi

echo "=== wheel tag check ==="
"$PYTHON" "$REPO_ROOT/tools/wheel_check.py" "${CHECK_ARGS[@]}"

WHEEL_BASENAME="$(basename "$WHEEL")"
SMOKE_HOST="$REPO_ROOT/scripts/wheel_smoke.py"

run_in_container() {
    local image="$1"
    local py="$2"
    local setup_cmds="$3"
    local label="$4"
    local cid=""

    echo
    echo "=== verify on $label ($image, platform linux/arm64) ==="

    cleanup() {
        if [ -n "$cid" ]; then
            docker rm -f "$cid" >/dev/null 2>&1 || true
        fi
    }
    trap cleanup EXIT

    cid="$(docker create --platform linux/arm64 "$image" sleep 3600)"

    if ! docker start "$cid" >/dev/null; then
        echo "failed to start $image under linux/arm64" >&2
        echo "register binfmt with:" >&2
        echo "  docker run --privileged --rm tonistiigi/binfmt --install arm64" >&2
        exit 1
    fi

    local arch_out
    if ! arch_out="$(docker exec "$cid" uname -m 2>&1)"; then
        echo "$arch_out" >&2
        if echo "$arch_out" | grep -qi 'exec format error'; then
            echo "binfmt/QEMU is not registered for arm64" >&2
            echo "register it with:" >&2
            echo "  docker run --privileged --rm tonistiigi/binfmt --install arm64" >&2
        fi
        exit 1
    fi
    echo "container uname -m: $arch_out"
    if [ "$arch_out" != "aarch64" ]; then
        echo "expected container architecture aarch64, got $arch_out" >&2
        exit 1
    fi

    docker cp "$WHEEL" "$cid:/tmp/$WHEEL_BASENAME"
    docker cp "$SMOKE_HOST" "$cid:/tmp/wheel_smoke.py"

    if ! docker exec "$cid" bash -lc "
        set -Eeuo pipefail
        $setup_cmds
        '$py' -m pip install --upgrade pip
        '$py' -m pip install --only-binary=:all: /tmp/$WHEEL_BASENAME
        '$py' /tmp/wheel_smoke.py
    "; then
        echo "verify failed inside $label" >&2
        exit 1
    fi

    cleanup
    trap - EXIT
}

# CPython 3.10 floor (abi3).
run_in_container \
    "python:3.10-slim" \
    "python3" \
    "true" \
    "CPython 3.10"

# Ubuntu 24.04 / CPython 3.12, the Raspberry Pi 5 interpreter.
# A venv avoids PEP 668 blocks on the system interpreter.
run_in_container \
    "ubuntu:24.04" \
    "/tmp/arco-venv/bin/python" \
    "export DEBIAN_FRONTEND=noninteractive
     apt-get update -qq
     apt-get install -y -qq python3 python3-pip python3-venv >/dev/null
     python3 -m venv /tmp/arco-venv" \
    "Ubuntu 24.04 / CPython 3.12"

echo
echo "=== verify_wheel_aarch64.sh PASSED ==="
sha256sum "$WHEEL"
