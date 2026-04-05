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

Before finishing **any** implementation task, all of the following must pass
locally. This applies to both human contributors and AI agents.

```bash
# 1. Unit tests — all green
python -m pytest tests/

# 2. Code formatting — zero issues
python -m black --check src/ tools/

# 3. Import ordering — zero issues
python -m isort --check-only src/ tools/

# 4. Examples — each script must complete without error
MPLBACKEND=Agg python tools/examples/astar_grid_obstacle.py --save /tmp/astar_grid.png
MPLBACKEND=Agg python tools/examples/astar_manhattan.py     --save /tmp/astar_manh.png
MPLBACKEND=Agg python tools/examples/astar_graph.py         --save /tmp/astar_graph.png
MPLBACKEND=Agg python tools/examples/astar_pipeline.py      --save /tmp/astar_pipeline.png
MPLBACKEND=Agg python tools/examples/rrt_planning.py        --save /tmp/rrt.png
MPLBACKEND=Agg python tools/examples/sst_planning.py        --save /tmp/sst.png
MPLBACKEND=Agg python tools/examples/ppp_planning.py        --save /tmp/ppp.png
MPLBACKEND=Agg python tools/examples/trajectory_optimization.py --save /tmp/traj.png
MPLBACKEND=Agg python tools/examples/route_planning.py      --save /tmp/route.png

# 5. Simulator smoke tests — each must record 5 s without error
#    (requires a real or virtual X display; use DISPLAY=:0 locally or
#     xvfb-run -a in CI)
SDL_AUDIODRIVER=dummy DISPLAY=:0 python tools/simulator/main/astar.py  --fps 30 --record /tmp/smoke_astar.mp4  --record-duration 5
SDL_AUDIODRIVER=dummy DISPLAY=:0 python tools/simulator/main/rrt.py    --fps 30 --record /tmp/smoke_rrt.mp4    --record-duration 5
SDL_AUDIODRIVER=dummy DISPLAY=:0 python tools/simulator/main/sst.py    --fps 30 --record /tmp/smoke_sst.mp4    --record-duration 5
SDL_AUDIODRIVER=dummy DISPLAY=:0 python tools/simulator/main/sparse.py --fps 30 --record /tmp/smoke_sparse.mp4 --record-duration 5
SDL_AUDIODRIVER=dummy DISPLAY=:0 python tools/simulator/main/ppp.py    --fps 30 --record /tmp/smoke_ppp.mp4    --record-duration 5
```

All GitHub workflow checks (push **and** release) must also pass before
opening or merging a pull request.
