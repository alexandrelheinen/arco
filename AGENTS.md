# AGENTS.md

This file provides guidance to AI coding agents (Antigravity, Gemini, Claude, Cursor, and other agents) when working with code in this repository.

@.guidelines/workflow/sdd.md
@.guidelines/workflow/integration.md
@.guidelines/workflow/tdd.md
@.guidelines/agents/writing.md
@.guidelines/style/naming.md
@.guidelines/languages/py.md
@.guidelines/languages/rs.md

Read these when the task touches a control loop, a planner, or numerical code, rather than every session:

- [.guidelines/style/defensive.md](.guidelines/style/defensive.md) — contracts, assertions, bounded resources, numerical rules
- [.guidelines/workflow/criticality.md](.guidelines/workflow/criticality.md) — how much rigor a module earns, and how a deviation is recorded

Read [CONTRIBUTING.md](CONTRIBUTING.md) for this project's development lifecycle, engineering standards, and CI gates. For ARCO's own project context, architecture invariants, and coding conventions, read [docs/guidelines.md](docs/guidelines.md). For architecture and layer documentation, read [docs/README.md](docs/README.md).

## Quick reference: Common tasks

| Task | Command |
|---|---|
| Run all gates before push | `bash scripts/validate.sh` |
| Fast gate check (skip coverage / deny) | `bash scripts/validate.sh --fast` |
| Run Python test suite | `pytest tests/ -v` |
| Run Rust test suite | `cargo nextest run --workspace` or `cargo test --workspace` |
| Format Python code | `python -m black --target-version py312 --line-length 79 src/ && python -m isort --line-length 79 src/` |
| Check Python formatting | `bash scripts/check_formatting.sh` |
| Rebuild Rust extension in place | `maturin develop` |
| Run simulator smoke test | `bash scripts/run_smoke_test.sh <scenario>` (`city`, `ppp`, `rrp`, `occ`) |

## Architecture at a glance

ARCO is a robotics path planning, guidance, and control library. The core algorithm layers are compiled Rust exposed to Python via PyO3 bindings, preserving backwards-compatible Python import paths and call syntax.

1. **Rust Workspace (`crates/`)**:
   - `arco-core`: Fundamental primitives, geometry, configuration parsing, seeded PCG64 RNG.
   - `arco-mapping`: Discrete grid structures (`ManhattanGrid`, `EuclideanGrid`), spatial graph hierarchy (`Graph` → `WeightedGraph` → `CartesianGraph` → `RoadGraph`), and KDTree occupancy.
   - `arco-planning`: Discrete planners (A*, Route), continuous sampling planners (RRT*, SST), trajectory optimizer, and pruner.
   - `arco-kinematics`: Planar kinematic models (RR, RRP).
   - `arco-control`: PID, Pure Pursuit, and MPCC (contouring model predictive control solved via Clarabel).
   - `arco-guidance`: Interpolation (B-spline, moving average) and exploration primitives (Dubins).
   - `arco-runtime`: In-memory thread-safe pub/sub message bus and pipeline runner.
   - `arco-testing`: Shared test fixtures and assertion helpers.
   - `arco-py`: PyO3 binding layer exposing native crates as `arco._arco`.

2. **Python Package (`src/arco/`)**:
   - Thin re-export facade mapping `arco._arco` native classes to historical module paths.
   - `arco.simulator`: Scenario simulations, OpenGL/Pygame visualization, still-frame generation (`arcosim`).

## Core Architecture Invariants

- **Maps are nouns** (`Grid`, `Occupancy`, `WeightedGraph`). They are passive data structures.
- **Planners are agent nouns with `-er` suffix** (`AStarPlanner`, `RRTPlanner`). They act on Maps.
- Planning algorithms accept a **map** object as their first argument.
- Shared engineering standards (SDD, TDD, V-cycle integration) in `.guidelines/` are authoritative.
