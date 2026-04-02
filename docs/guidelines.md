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

Run **both** formatters before every commit:

```bash
python -m black --target-version py312 src/ tests/
python -m isort src/ tests/
```

- `black` target version: `py312`.
- `isort` default profile (no extra configuration needed).

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
