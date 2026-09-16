#!/usr/bin/env bash
# scripts/validate.sh
#
# The single validation gate.  CI runs this script and nothing else, so
# "it passed locally" and "CI is green" mean the same thing.  See
# .guidelines/workflow/integration.md.
#
# Usage: bash scripts/validate.sh [--fast]
#   --fast  skip coverage and cargo-deny, for a tight edit loop
#
# Exit code: 0 = all gates pass, non-zero = the first blocking failure.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

FAST=false
[ "${1:-}" = "--fast" ] && FAST=true

# Coverage floor per FR-TEST-03.  Lines, regions and functions; there is
# no branch-coverage gate in Rust today, see ADR-007.
COVERAGE_FLOOR="${ARCO_COVERAGE_FLOOR:-80}"

PYTHON="${ARCO_PYTHON:-$REPO_ROOT/.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON="$(command -v python3)"

export PATH="$HOME/.cargo/bin:$PATH"

step() { printf '\n=== %s ===\n' "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }

require() {
    if ! have "$1"; then
        echo "missing required tool: $1" >&2
        echo "install it with: $2" >&2
        exit 127
    fi
}

require cargo "https://rustup.rs"

# ---------------------------------------------------------------- Rust ----
step "cargo fmt"
cargo fmt --all --check

step "cargo clippy"
cargo clippy --all-targets --all-features --workspace -- -D warnings

step "cargo test"
# --no-tests=pass keeps an empty crate from failing the gate.  A crate
# that should have tests is caught by the coverage floor, not by this.
if have cargo-nextest; then
    cargo nextest run --workspace --all-features --no-tests=pass
else
    cargo test --workspace --all-features
fi
cargo test --workspace --doc

step "cargo doc"
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features --workspace

if [ "$FAST" = false ]; then
    step "cargo deny"
    if have cargo-deny; then
        cargo deny check
    else
        echo "cargo-deny not installed, skipping (install: cargo install cargo-deny)"
    fi
fi

# ------------------------------------------------------- Rust + Python ----
# Coverage has to instrument the extension before maturin builds it, or a
# pytest run reports into a different profile and the report comes back
# empty.  This is the show-env flow, and it is why the open question about
# PyO3 coverage in docs/rust/SPEC.md is closed.
if [ "$FAST" = false ] && have cargo-llvm-cov; then
    step "coverage (rust + python, floor ${COVERAGE_FLOOR}%)"
    # shellcheck disable=SC1090
    source <(cargo llvm-cov show-env --sh)
    cargo llvm-cov clean --workspace
    cargo build --workspace --all-features
    # The Rust suite runs again, instrumented this time.  Without it the
    # report covers only what a pytest run reaches through the extension
    # module, which is a fraction of the workspace and makes the floor
    # mean nothing for a crate the bindings do not expose yet.
    if have cargo-nextest; then
        cargo nextest run --workspace --all-features --no-tests=pass
    else
        cargo test --workspace --all-features
    fi
    # Build through the target interpreter so the extension lands in the
    # environment pytest will use.  Failing loudly matters: a silent skip
    # here measures coverage against a stale build.
    if "$PYTHON" -c "import maturin" >/dev/null 2>&1; then
        "$PYTHON" -m maturin develop --quiet
    elif have maturin; then
        maturin develop --quiet
    else
        echo "maturin is not installed for $PYTHON" >&2
        echo "install it with: $PYTHON -m pip install 'maturin>=1.7,<2'" >&2
        exit 127
    fi
    "$PYTHON" -m pytest tests/ -q
    cargo llvm-cov report \
        --fail-under-lines "$COVERAGE_FLOOR" \
        --fail-under-regions "$COVERAGE_FLOOR" \
        --fail-under-functions "$COVERAGE_FLOOR"
else
    step "python tests"
    "$PYTHON" -m pytest tests/ -q
fi

# -------------------------------------------------------------- Python ----
step "python formatting"
ARCO_PYTHON="$PYTHON" bash scripts/check_formatting.sh

printf '\n=== validate.sh PASSED ===\n'
