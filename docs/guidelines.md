# ARCO Coding Guidelines

This document is the single authoritative reference for coding conventions in the ARCO project. All contributors and AI agents should follow these rules rigorously — they are **not optional**.

---

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

---

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

---

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

---

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

---

## 5. Type Annotations

Variable and parameter typing is **strongly enforced**. Every public method signature must include full type annotations. Use `from __future__ import annotations` at the top of each file.

---

## 6. Testing

- Tests must be defined at the same time as the structures they cover (V-cycle). They should be created before moving on to the next task.
- Place tests in `tests/` mirroring the `src/arco/` layout.
- Use `pytest`. Run with:
  ```bash
  pytest tests/
  ```
- Stub/not-yet-implemented methods should be marked `@pytest.mark.xfail(strict=True, raises=NotImplementedError)` rather than skipped.

---

## 7. Configuration Parameters

Tunable algorithm parameters belong in `.yml` files under `config/`. The YAML structure should be human-readable and does not have to mirror the class hierarchy.

---

## 8. Architecture Invariants

- Planning algorithms accept a **map** object as their first argument.
  - Grid-based planners (`AStarPlanner`, `DStarPlanner`) require a `Grid` subclass.
  - Sampling-based planners (`RRTPlanner`, `SSTPlanner`) require an `Occupancy` subclass.
- The **guidance** layer is applied after planning; it handles interpolation (B-splines) and exploration primitives (Dubins, Reeds-Shepp) for RRT-family algorithms.
- The `AStarPlanner` uses `graph.heuristic` (Euclidean distance) as the default heuristic, not `graph.distance` (Manhattan). This prevents L-shaped paths on symmetric Manhattan grids.

---

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
