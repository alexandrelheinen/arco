# Contributing to ARCO

Thank you for contributing.

## Prerequisites

- Python 3.10+
- A Rust toolchain. `rust-toolchain.toml` pins the version, so
  [rustup](https://rustup.rs) installs the right one on first build.
- Git

## Setup

```bash
git clone --recurse-submodules https://github.com/alexandrelheinen/arco.git
cd arco
pip install -e ".[dev]"
```

The build backend is [maturin](https://www.maturin.rs), which compiles the
Rust workspace under `crates/` and installs the extension beside the
Python package as `arco._arco`. An editable install rebuilds the extension
when Rust sources change; `maturin develop` forces a rebuild.

The gates need four cargo tools:

```bash
cargo install cargo-nextest cargo-llvm-cov cargo-deny --locked
```

## Development Workflow

1. Create a feature branch from main.
2. Implement the change with focused commits.
3. Add or update tests in the mirrored tests layout.
4. Run local validation before opening a pull request.

## Local Validation

One script runs every gate, and CI runs the same script, so "it passed
locally" and "CI is green" mean the same thing:

```bash
bash scripts/validate.sh
```

It runs `cargo fmt`, `cargo clippy` at deny-warnings, the Rust tests and
doc tests, `cargo doc`, `cargo deny`, then builds the extension under
coverage instrumentation and runs `pytest` through it, and finishes with
the Python formatting gate. Pass `--fast` to skip coverage and the
dependency audit during a tight edit loop.

Coverage instruments the extension before maturin builds it, which is why
the script sources `cargo llvm-cov show-env` rather than calling
`cargo llvm-cov` directly. Running `pytest` against an uninstrumented
build reports an empty profile.

Read [docs/rust/STYLE.md](docs/rust/STYLE.md) before writing Rust.

## Coding Rules

The authoritative coding standard is [docs/guidelines.md](docs/guidelines.md).

All contributions, including AI-assisted changes, must follow this file.

## Pull Request Checklist

- Behavior changes are covered by tests
- Public APIs are typed and documented (Google-style docstrings)
- Formatting and imports are clean
- Diff is scoped to the requested task

## Documentation

Update documentation when adding or changing behavior:

- [README.md](README.md) for user-facing behavior
- [docs/PLANNING.md](docs/PLANNING.md) and algorithm notes for planning changes
- [docs/ROADMAP.md](docs/ROADMAP.md) for future work updates
