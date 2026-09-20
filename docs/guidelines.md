# ARCO Coding Guidelines

This document covers ARCO-specific conventions that are not part of the
shared engineering guidelines. Generic Python style (formatting, typing,
docstrings, package layout, testing process) lives in
[.guidelines/languages/py.md](../.guidelines/languages/py.md); general
naming rules live in [.guidelines/style/naming.md](../.guidelines/style/naming.md).
This file only documents where ARCO diverges from or adds to those
defaults.

## 1. Naming Conventions

- **Maps are nouns** (`Grid`, `Occupancy`, `WeightedGraph`). They are passive data structures.
- **Planners are agent nouns with `-er` suffix** (`AStarPlanner`, `RRTPlanner`). They act on Maps.
- Config files (`.yml`) in the `config/` folder may use a flat structure tailored for human readability; they do not need to mirror the code architecture.

## 2. Code Formatting

Formatting is enforced on **Python package code** (`src/`).
Test files (`tests/`) are excluded — they are not production code and do not
need to be perfectly formatted or documented.

Run **both** formatters before committing Python changes:

```bash
python -m black --target-version py312 --line-length 79 src/
python -m isort --line-length 79 src/
```

- `black` target version: `py312`, line length: `79`. This is an accepted
  per-project override of [.guidelines/languages/py.md](../.guidelines/languages/py.md),
  which defaults to line-length 88.
- `isort` default profile (no extra configuration needed).
- CI enforces these rules on `src/` via `scripts/check_formatting.sh`.

## 3. Testing

Tests are placed in `tests/`, mirroring the `src/arco/` layout. Stub or
not-yet-implemented methods are marked
`@pytest.mark.xfail(strict=True, raises=NotImplementedError)` rather than
skipped, per [.guidelines/workflow/tdd.md](../.guidelines/workflow/tdd.md).

## 4. Configuration Parameters

Tunable algorithm parameters belong in `.yml` files under `config/`. The YAML structure should be human-readable and does not have to mirror the class hierarchy.

## 5. Architecture Invariants

- Planning algorithms accept a **map** object as their first argument.
  - Grid-based planners (`AStarPlanner`, `DStarPlanner`) require a `Grid` subclass.
  - Sampling-based planners (`RRTPlanner`, `SSTPlanner`) require an `Occupancy` subclass.
- A*, RRT*, and SST inherit :class:`~arco.planning.cost.PlannerCost` (via
  `DiscretePlanner` / `ContinuousPlanner`).  Override `distance` and/or
  `heuristic` to customize path cost without rewriting the search loop.
- Optional policy hooks (`sampler=`, `steerer=`, `segment_free=`, `cost=`,
  telemetry `publisher=`, optimizer `cost_terms=`, tracking avoidance)
  always default to historical behavior.  See [PLANNING.md](PLANNING.md)
  § Extension Points and `arco.protocols`.
- The **guidance** layer provides interpolation and exploration primitives.
  `ExplorationPrimitive` is **not** auto-wired into RRT/SST; inject it via
  `steerer=` when kinodynamic expansion is required.
- The `AStarPlanner` uses `graph.heuristic` (Euclidean distance) as the default heuristic, not `graph.distance` (Manhattan). This prevents L-shaped paths on symmetric Manhattan grids.

## 6. Spatial Graph Hierarchy

ARCO separates generic graph topology from spatial geometry through a three-level hierarchy:

```
Graph                  — pure topology (nodes + edges, no weights)
  └─ WeightedGraph     — adds numeric edge weights; no positional data
       └─ CartesianGraph — adds N-dimensional Cartesian node positions
            └─ RoadGraph — adds per-edge geometry waypoints
```

- **`WeightedGraph`** is fully generic: `add_node(id)`, `add_edge(a, b, weight)`.
  It has no concept of position, distance between nodes, or spatial queries.
- **`CartesianGraph`** extends `WeightedGraph` with N-dimensional positions stored as
  `numpy.ndarray`. Node positions are added via `add_node(id, *coords)`.
  Edge weights default to the Euclidean distance between endpoint positions.
  Provides `heuristic()`, `find_nearest_node()`, and `project_to_nearest_edge()`.
- **`Grid`** subclasses expose a `position(cell_idx)` method that computes the
  Cartesian position of a cell from its index and `cell_size`. The heuristic uses this
  method, so it correctly accounts for non-unit cell sizes.

### N-dimensional design rules

- All node positions in `CartesianGraph` and `Grid` are `numpy.ndarray` objects.
- `find_nearest_node(position: np.ndarray)` and
  `project_to_nearest_edge(position: np.ndarray)` accept position arrays
  of any dimension N.
- `RouteRouter.plan(start_position: np.ndarray, goal_position: np.ndarray)`
  accepts position arrays of any dimension N.
- `RouteResult.start_projection` and `goal_projection` are `numpy.ndarray`.
- Planners or methods that only support a specific dimension should raise
  `ValueError` if given a graph or position of the wrong dimension.

---

## 7. Pre-flight Checklist

Install the git pre-push hook once per clone to block pushes that fail
formatting or unit tests:

```bash
bash scripts/install_hooks.sh
```

The hook (`hooks/pre-push`) runs formatting and unit tests automatically on
every `git push`. Before pushing **any** branch, also run smoke tests to
catch simulator-level import/runtime issues:

```bash
bash scripts/check_formatting.sh
bash scripts/run_tests.sh
# Requires xvfb + ffmpeg:
for s in city occ ppp rrp; do
  bash scripts/run_smoke_test.sh "$s"
done
```

| Gate | Script | What it checks |
|------|--------|----------------|
| 0 | `scripts/validate.sh` | Every gate below, plus the Rust workspace. CI runs this and nothing else |
| 1 | `scripts/check_formatting.sh` | `black` + `isort` (blocking), `pydocstyle` (warning) |
| 2 | `scripts/run_tests.sh` | `pytest` unit tests |
| 3 | `scripts/run_smoke_test.sh <scenario>` | short headless `arcosim` recording |
| 4 | `scripts/generate_videos.sh` | full-length simulation videos (release) |

> **⚠️ Do not skip smoke tests.**  
> They are the only local gate that imports and executes every simulator
> module. Skipping them is how import-time errors escape into CI.

All GitHub workflow checks (push **and** release) must pass before a pull
request is merged.  If a workflow fails, investigate with GitHub MCP tools
before concluding the session.

---

## 8. Shared Configuration Files — Consumer Audit Rule

Whenever a **shared configuration file** (any file under `src/arco/config/`,
including `colors.yml`, `astar.yml`, etc.) is **restructured** (keys renamed,
sections added/removed), you **must** audit every consumer before pushing.

### Procedure

1. **Identify all consumers** of the file being changed:

   ```bash
   # For colors.yml — find every file that reads it, directly or indirectly
   grep -rn 'load_config("colors")\|from arco.config.palette' src/ tests/ \
       | grep -v "\.pyc"
   ```

   > For other config files replace `"colors"` with the filename being
   > modified.

2. **Verify each consumer** is compatible with the new structure.  A consumer
   is compatible when **all key paths it accesses exist** in the new file.  If
   any consumer still references a deleted or renamed key, update it now.

3. **Confirm with a quick import check** after updating:

   ```bash
   python -c "
   import importlib, sys
   modules = [
       'arco.config.palette',
       'arco.simulator.main.city',
       'arco.simulator.main.ppp',
       'arco.simulator.main.rrp',
       'arco.simulator.main.occ',
       'arco.simulator.scenes.sparse',
   ]
   for m in modules:
       importlib.import_module(m)
       print('OK', m)
   "
   ```

4. Run the **full pre-flight checklist** including smoke tests (gate 4) to
   confirm that no module-level `KeyError` or `ImportError` has been
   introduced.

### Why this rule exists

After restructuring any shared config, zero files may reference a key path
that no longer exists.  The import check above catches this at module-load
time, before any simulation runs.

## 9. Tests That Import Display-Only Modules

Some simulator entry points (e.g. `ppp.py`, `rrp.py`, `occ.py`)
import `pygame` and/or `OpenGL` at module level.  These modules are **not**
installed in the CI test-runner environment (which has no display libraries).

### Rule

Any test file that needs to import from a pygame/OpenGL simulator entry
point **must** start with:

```python
pygame = pytest.importorskip("pygame")
```

placed **before** the import that triggers `pygame`.  This causes pytest to
skip the entire test module with a clear message when pygame is unavailable,
rather than crashing the collection phase with `ModuleNotFoundError`.

## 10. Example Functions: Data-Series Dimensions

When plotting multi-step simulations (e.g. Lyapunov functions, V(t)):

- **time axis** and **value axis** must always have the same length.
- If the simulation runs for `N` steps with a fixed `dt`, the time axis is
  `np.arange(N) * dt` — *not* a cumulative sum of per-waypoint optimizer
  durations (which has length equal to the number of waypoints, not steps).

## 11. Adding a Required Constructor Parameter — All Call-Sites Rule

When a constructor parameter is changed from optional to **required** (i.e. it
gains no default value), every call site in the entire codebase — including
simulator `scenes/`, standalone `examples/`, tests, and any other consumer —
**must** be updated in the same commit.

> After changing any constructor or public-method signature, run:
> ```bash
> grep -rn "ClassName(" src/ tests/
> ```
> and verify *every* hit is updated.  Then run smoke tests
> (`scripts/run_smoke_test.sh <scenario>` for each scenario) to confirm no
> runtime crashes survive.

This check is **mandatory** before any commit that touches a public API.

## 12. Rust

ARCO's algorithm core is compiled Rust behind PyO3 bindings, so the
Python package keeps its historical import paths and call syntax. See
[docs/rust/SPEC.md](rust/SPEC.md) for the scope and acceptance criteria,
[docs/rust/PLAN.md](rust/PLAN.md) for the order of work, and
[docs/decisions.md](decisions.md) for why the approach was chosen.

The authoritative Rust coding standard is
[.guidelines/languages/rs.md](../.guidelines/languages/rs.md).
[docs/rust/STYLE.md](rust/STYLE.md) stands to it as this file stands to
[.guidelines/languages/py.md](../.guidelines/languages/py.md): ARCO
deltas only, nothing the shared file already covers.

Two shared files govern the parts of ARCO where a wrong answer costs more
than a rerun. [.guidelines/style/defensive.md](../.guidelines/style/defensive.md)
holds the contract, assertion, bounded-resource and numerical rules, and
[.guidelines/workflow/criticality.md](../.guidelines/workflow/criticality.md)
says how much of that a given module owes, with the procedure for
recording a deviation. ARCO assigns a criticality level per crate in
[docs/rust/STYLE.md](rust/STYLE.md#1-criticality-per-crate).

Rules from sections 1 through 11 above that describe the domain rather
than the language carry over unchanged: maps are nouns, planners take the
`-er` suffix, planners accept a map as their first argument, and the
graph hierarchy keeps its four levels. Rules that describe Python
mechanics do not carry over; every such adaptation is recorded in
[docs/rust/DEVIATIONS.md](rust/DEVIATIONS.md).

Run the same gate CI runs before pushing any branch that touches Rust:

```bash
bash scripts/validate.sh
```
