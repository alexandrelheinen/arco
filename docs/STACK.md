# Tech Stack

## Python version

ARCO requires **Python 3.10 or later**. The package is tested on Python 3.10
and formatted to the 3.12 target via Black.

## Core dependencies

These are installed automatically when you install ARCO:

| Package | Role |
|---------|------|
| `numpy` | Array math, spatial operations, trajectory representation |
| `scipy` | KDTree nearest-neighbor queries (occupancy collision checks) |

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

```bash
git clone https://github.com/alexandrelheinen/arco.git
cd arco
pip install -e ".[dev]"
```

For the visualization tools:

```bash
pip install -e ".[tools]"         # static image mode only
pip install -e ".[tools,pygame]"  # full arcosim (real-time + static)
```

## Running the test suite

```bash
pytest tests/ -v
```

All tests are in `tests/`, mirroring the `src/arco/` package structure.

## Formatting

```bash
python -m black --target-version py312 --line-length 79 src/
python -m isort --line-length 79 src/
```

## Local CI validation

The master local validation script runs all required CI gates:

```bash
bash scripts/pre_push.sh
```

Individual gates:

| Script | What it checks |
|--------|----------------|
| `scripts/check_formatting.sh` | black + isort (blocking), pydocstyle (warning) |
| `scripts/run_tests.sh` | pytest unit tests |
| `scripts/run_examples.sh` | `arcosim --image` headless image generation |
| `scripts/run_smoke_tests.sh` | arcosim short headless recordings |
| `scripts/generate_videos.sh` | arcosim full-length simulation videos |
