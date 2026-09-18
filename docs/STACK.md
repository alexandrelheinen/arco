# Tech Stack

## Python version

ARCO requires **Python 3.10 or later**. The package is tested on Python 3.10
and formatted to the 3.12 target via Black.

## Rust toolchain

The algorithms are compiled Rust reached through PyO3 bindings, published as
the `arco._arco` extension module. A wheel carries that module already built
and needs no toolchain on the installing machine. Building from source needs
the compiler pinned in [rust-toolchain.toml](../rust-toolchain.toml), which
`rustup` installs on the first build, and `maturin` as the build backend.

The extension targets the stable ABI from Python 3.10 upward, so one wheel
per platform serves every supported interpreter.

## Core dependencies

These are installed automatically when you install ARCO:

| Package | Role |
|---------|------|
| `numpy` | Array math at the binding boundary, trajectory representation |
| `pyyaml` | Scenario and configuration loading |
| `scipy` | Scene generation in the simulator and in the illustration tools |

Nearest-neighbor queries no longer go through `scipy`. The occupancy tree is
native, which is why a query is faster and a tree build is slower than the
`cKDTree` it replaced; both numbers are in the port's validation write-up.

## Optional dependency groups

### `dev` — development and testing

```bash
pip install arco[dev]
```

| Package | Role |
|---------|------|
| `pytest` | Unit test runner |
| `black` | Code formatter (line length 79, target py312) |
| `isort` | Import sorter (black-compatible profile) |
| `matplotlib` | Plot generation in example scripts |
| `pyyaml` | YAML config loading |
| `pylint` | Static analysis |
| `pydocstyle` | Docstring style checker (Google convention) |

### `tools` — visualization tools (arcosim)

```bash
pip install arco[tools]        # static image mode only (matplotlib)
pip install arco[tools,pygame] # full arcosim (adds pygame + PyOpenGL)
```

| Package | Role |
|---------|------|
| `matplotlib` | Static figure generation (`arcosim --image`) |
| `pyyaml` | Scenario YAML loading |
| `pygame >= 2.0` | Real-time simulation window (arcosim) |
| `PyOpenGL >= 3.1` | OpenGL rendering in arcosim |

## Installation

Installing a wheel needs nothing but Python:

```bash
pip install arco
```

Building from a checkout compiles the workspace, so it needs the Rust
toolchain described above:

```bash
git clone https://github.com/alexandrelheinen/arco.git
cd arco
pip install -e ".[dev]"
```

During development, `maturin` rebuilds the extension in place without
reinstalling the package, which is the loop to use after a Rust change:

```bash
python -m maturin develop
```

For the visualization tools:

```bash
pip install -e ".[tools]"         # static image mode only
pip install -e ".[tools,pygame]"  # full arcosim (real-time + static)
```

## Running the test suite

One script runs every gate, and CI runs that script and nothing else:

```bash
bash scripts/validate.sh
```

The two suites separately, when a full gate run is more than the change
needs:

```bash
pytest tests/ -v
cargo nextest run
```

Python tests are in `tests/`, mirroring the `src/arco/` package structure.
Rust tests live beside the code they cover and in each crate's own `tests/`
directory. A Rust change needs `maturin develop` before the Python suite
sees it.

## Formatting

```bash
python -m black --target-version py312 --line-length 79 src/
python -m isort --line-length 79 src/
```

## Local CI validation

Run the same gates CI uses:

| Script | What it checks |
|--------|----------------|
| `scripts/validate.sh` | Every gate below, plus the Rust lints, doc tests, dependency audit and coverage floors. This is what CI runs |
| `scripts/check_formatting.sh` | black + isort (blocking), pydocstyle (warning) |
| `scripts/run_tests.sh` | pytest unit tests |
| `scripts/run_smoke_test.sh <scenario>` | short headless `arcosim` recording |
| `scripts/generate_videos.sh` | full-length simulation videos (`--release` for CI) |

```bash
bash scripts/validate.sh
```

The smoke scenarios are the map files in `map/`, and a recording runs the
planner to completion before the first frame, so `occ` takes minutes rather
than the seconds its recording length suggests:

```bash
for s in city occ ppp rrp; do
  bash scripts/run_smoke_test.sh "$s"
done
```
