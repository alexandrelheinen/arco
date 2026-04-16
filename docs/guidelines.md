# ARCO Coding Guidelines

This document is the single authoritative reference for coding conventions in the ARCO project. All contributors and AI agents should follow these rules rigorously — they are **not optional**.



## 1. File and Package Structure

- **One main class per `.py` file.** Small, tightly-coupled auxiliary classes (e.g. nested `Node`/`Edge` inside `Graph`) may coexist in the same file using nesting to reproduce namespace effects (e.g. `mapping.Graph.Node`).
- **Use folders with `__init__.py` instead of `_`-separated suffixes.** If several files share a common suffix (e.g. `oriented_graph.py`, `weighted_graph.py`), turn that suffix into a sub-package and drop it from the filenames:
  ```
  mapping/graph/
      __init__.py     ← re-exports Graph, OrientedGraph, WeightedGraph
      base.py         ← Graph
      oriented.py     ← OrientedGraph
      weighted.py     ← WeightedGraph
  ```
  The public import path then becomes `arco.mapping.graph.oriented`, `arco.mapping.graph.weighted`, etc.
- **`__init__.py` files are re-export-only.** They must not contain class or function definitions; they only import and expose the public API of the package.
- Add sub-folders to better group classes that depend on each other or are closely related.



## 2. Naming Conventions

- **Maps are nouns** (`Grid`, `Occupancy`, `WeightedGraph`). They are passive data structures.
- **Planners are agent nouns with `-er` suffix** (`AStarPlanner`, `RRTPlanner`). They act on Maps.
- **No `_`-separated suffixes when a sub-package can replace them** (see §1 above).
- Config files (`.yml`) in the `config/` folder may use a flat structure tailored for human readability; they do not need to mirror the code architecture.

### Physical Variable Naming

All physical variables must follow the `who_what` (or `who_what_where` when frame of reference matters) convention. The rules below are **mandatory** for all layers.

**Physical quantity, not unit.** Never suffix a variable with its SI unit. Use the quantity name instead.

| Wrong | Correct |
|---|---|
| `size_m` | `physical_size` or `cell_size` |
| `radius_m` | `lookahead_distance` |
| `speed_ms` | `max_speed` |

Non-SI config parameters are explicitly exempt: a configuration value expressed in deg/s for human readability may keep its unit suffix (e.g. `max_turn_rate_deg_s` in a YAML file). The exemption must be documented with a comment in the config file.

**Qualifiers are prefixes.** `max`, `min`, `avg`, and similar qualifiers always come first.

| Wrong | Correct |
|---|---|
| `speed_max` | `max_speed` |
| `value_avg` | `avg_value` |

**Integers are suffixed with `_count`.** Never use a `num_` prefix or leave an integer qualifier implicit.

| Wrong | Correct |
|---|---|
| `num_obstacles` | `obstacle_count` |
| `waypoints_per_edge` | `waypoints_per_edge_count` |
| `num_intersections` | `intersection_count` |

**Lists use the plural form**, unless the list represents a coordinate sequence.

| Wrong | Correct |
|---|---|
| `ring_radius` (list of radii) | `ring_radii` |

**Two-word quantity names are valid** and should not be collapsed. `turn_rate`, `cell_size`, and `obstacle_fraction` are all acceptable base names.



## 3. Documentation — Google Style

All public classes and methods must have Google-style docstrings with the appropriate sections.

```python
def plan(self, start: Any, goal: Any) -> Optional[List[Any]]:
    """Plan a path from start to goal.

    Args:
        start: The start node.
        goal: The goal node.

    Returns:
        A list of nodes from start to goal, or None if no path exists.

    Raises:
        NotImplementedError: Subclasses must implement this method.
    """
```

Required sections:
- `Args` — for every parameter (skip only when there are none).
- `Returns` — for every non-`None` return value.
- `Raises` — whenever the method raises an exception intentionally.
- `Yields` — for generator methods.

References: [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html), [PEP 8](https://peps.python.org/pep-0008/).



## 4. Code Formatting

Formatting is enforced on **production code only** (`src/` and `tools/`).
Test files (`tests/`) are excluded — they are not production code and do not
need to be perfectly formatted or documented.

Run **both** formatters before every commit:

```bash
python -m black --target-version py312 --line-length 79 src/ tools/
python -m isort --line-length 79 src/ tools/
```

- `black` target version: `py312`, line length: `79`.
- `isort` default profile (no extra configuration needed).
- CI enforces these rules on `src/` and `tools/` only.



## 5. Type Annotations

Variable and parameter typing is **strongly enforced**. Every public method signature must include full type annotations. Use `from __future__ import annotations` at the top of each file.



## 6. Testing

- Tests must be defined at the same time as the structures they cover (V-cycle). They should be created before moving on to the next task.
- Place tests in `tests/` mirroring the `src/arco/` layout.
- Use `pytest`. Run with:
  ```bash
  pytest tests/
  ```
- Stub/not-yet-implemented methods should be marked `@pytest.mark.xfail(strict=True, raises=NotImplementedError)` rather than skipped.

An imperative order (do, implement, make, add...) is not only about writting the code. It must include all the V-cycle.

### 6.1. Respect the V-cycle

As indicated above, all work of an AI must include all the descending and ascending steps of the cycle. For each row numerated below, the two actions (descend and ascend) must be done **at the same time**:

1. Add a documentation of the work, feature, but... Use Github issues if possible, otherwise, go diretly into the Github PR and document every step in comments. It includes: goal/objectives and acceptable criteria
2. Implement the architecture of the code (classes, public interface, file organization, dependencies) and the (functional) unit tests at the same time. The testing must come first then coding: the performance of the algorithm is independent of its implementation. By reading the acceptance criteria above, you must already know which values to expect.
3. Do the coding. This is the 3-rd step: Fill the stubs lefted by the architecture definition. Implement algorithms, data structure and private/local utilities. Add unit testing for private functions as well (fine testing/non-functional tests).
4. Then, run the tests. Ideally proving 100% coverage (at least 90% would be great!). If this step fails, go back to step 2: Review your architecture, your functional tests, and go back to the cycle.
5. Implement high level simulations if all the testing are passing. Add visual inspection (either images or videos) in the `tools` folder. Add material for the presentation and documentation of the tool. Add the appropriate documentation of the newly implemented feature, of fix the lines affected by the changes. All github workflows must pass: both at push and release! If some is wrong in this step, go back to step number 1.

This complete the V-cycle. Once the acceptance criteria are met and all the Github workflows (autotests) are passing (both at push and release, test them all locally or add the tooling to test it), you can push your branch and trigger the review.



## 7. Configuration Parameters

Tunable algorithm parameters belong in `.yml` files under `config/`. The YAML structure should be human-readable and does not have to mirror the class hierarchy.



## 8. Architecture Invariants

- Planning algorithms accept a **map** object as their first argument.
  - Grid-based planners (`AStarPlanner`, `DStarPlanner`) require a `Grid` subclass.
  - Sampling-based planners (`RRTPlanner`, `SSTPlanner`) require an `Occupancy` subclass.
- The **guidance** layer is applied after planning; it handles interpolation (B-splines) and exploration primitives (Dubins, Reeds-Shepp) for RRT-family algorithms.
- The `AStarPlanner` uses `graph.heuristic` (Euclidean distance) as the default heuristic, not `graph.distance` (Manhattan). This prevents L-shaped paths on symmetric Manhattan grids.



## 9. Spatial Graph Hierarchy

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

## 10. Language — US English Only

All source code identifiers, comments, docstrings, documentation (Markdown),
configuration files (YAML/JSON), and commit messages **must use American
English** spelling. This rule applies to all human contributors and AI agents
without exception.

Common corrections:

| Use | Not |
|-----|-----|
| color | colour |
| behavior | behaviour |
| initialize / initialized | initialise / initialised |
| optimize / optimized / optimizer | optimise / optimised / optimiser |
| center / centered / centerline | centre / centred / centreline |
| neighborhood | neighbourhood |
| meters | metres |
| normalize / normalized | normalise / normalised |
| recognize / recognized | recognise / recognised |
| analyze | analyse |
| finalize | finalise |
| discretize / discretized | discretise / discretised |
| visualize | visualise |
| utilize | utilise |
| minimize | minimise |
| maximize | maximise |
| realize / realizability | realise / realisability |
| nearest-neighbor | nearest-neighbour |
| deserializes | deserialises |

**Exception:** External library API parameters that use UK spelling (e.g.
`tqdm(colour=...)`) must be left unchanged to avoid breaking calls.

## 11. Pre-flight Checklist

Install the git pre-push hook once per clone to block pushes that fail
formatting or unit tests:

```bash
bash scripts/install_hooks.sh
```

The hook (`hooks/pre-push`) runs gates 1 and 2 automatically on every
`git push`.  Before pushing **any** branch, also run the full validation
script to catch simulator-level issues:

```bash
# Full validation (requires xvfb + ffmpeg for smoke tests and videos)
bash scripts/pre_push.sh
```

`scripts/pre_push.sh` runs **all five CI gates** in the same order as the
GitHub workflows.  All gates are mandatory — none may be skipped:

| Gate | Script | What it checks |
|------|--------|----------------|
| 1 | `scripts/check_formatting.sh` | `black` + `isort` (blocking), `pydocstyle` (warning) |
| 2 | `scripts/run_tests.sh` | `pytest` unit tests |
| 3 | `scripts/run_examples.sh` | `arcosim --image` headless image generation for every scenario |
| 4 | `scripts/run_smoke_tests.sh` | `arcosim` short headless recording for every simulator |
| 5 | `scripts/generate_videos.sh` | `arcosim` full-length simulation videos |

> **⚠️ Do not skip any gate.**  
> The smoke tests are the only local gate that imports and executes every
> simulator module.  Skipping them is how import-time errors in simulator
> files escape the local validation loop and fail in CI.

All GitHub workflow checks (push **and** release) must pass before a pull
request is merged.  If a workflow fails, investigate with GitHub MCP tools
before concluding the session.

---

## 12. Shared Configuration Files — Consumer Audit Rule

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
       'arco.tools.simulator.main.city',
       'arco.tools.simulator.main.vehicle',
       'arco.tools.simulator.main.rr',
       'arco.tools.simulator.scenes.sparse',
       'arco.tools.simulator.scenes.rrt',
       'arco.tools.simulator.scenes.sst',
       'arco.tools.simulator.scenes.astar',
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


## 13. Tests That Import Display-Only Modules

Some simulator entry points (e.g. `ppp.py`, `rrp.py`, `rr.py`, `occ.py`)
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


## 14. Example Functions: Data-Series Dimensions

When plotting multi-step simulations (e.g. Lyapunov functions, V(t)):

- **time axis** and **value axis** must always have the same length.
- If the simulation runs for `N` steps with a fixed `dt`, the time axis is
  `np.arange(N) * dt` — *not* a cumulative sum of per-waypoint optimizer
  durations (which has length equal to the number of waypoints, not steps).


## 15. Adding a Required Constructor Parameter — All Call-Sites Rule

When a constructor parameter is changed from optional to **required** (i.e. it
gains no default value), every call site in the entire codebase — including
simulator `scenes/`, standalone `examples/`, tests, and any other consumer —
**must** be updated in the same commit.

> After changing any constructor or public-method signature, run:
> ```bash
> grep -rn "ClassName(" src/ tests/
> ```
> and verify *every* hit is updated.  Then run `scripts/pre_push.sh` (or at
> minimum `scripts/run_smoke_tests.sh`) to confirm no runtime crashes survive.

This check is **mandatory** before any commit that touches a public API.


## 16. pyreverse — Module Name Conflicts with `__init__` Re-exports

`pyreverse` (pylint ≤ 3.x) crashes with `KeyError: 'arco.X.module'` when a
package's `__init__.py` contains a `from .module import …` statement **and**
the module name is not pre-registered in pyreverse's internal `module_info`
table.  This tends to occur for top-level modules (not sub-packages) added to
packages that use `__init__.py` as a re-export hub.

### Prevention rule

Whenever a new **module file** (not a sub-package) is added to a package whose
`__init__.py` re-exports it, add `--ignore=<filename>.py` to both `pyreverse`
invocations in `.github/workflows/generate_images.yml`.  Alternatively,
restructure the new module as a sub-package (a folder with its own
`__init__.py`) so pyreverse handles it like a package, not a module.
