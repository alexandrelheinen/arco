# ARCO Public API Failure Modes

Contracts for invalid inputs and unsatisfiable planning problems on the
public interface documented in [API.md](API.md).

Legend:

- **Raise** — caller must catch or avoid
- **Sentinel** — returns `None` (or empty diagnostics) without raising
- **Warn** — emits `DeprecationWarning` and continues

---

## Mapping

| API | Failure | Behavior | Tests |
|-----|---------|----------|-------|
| `KDTreeOccupancy(points, clearance)` | empty `points` | Raise `ValueError` | `tests/mapping/` |
| `KDTreeOccupancy(points, clearance)` | `clearance <= 0` | Raise `ValueError` | `tests/mapping/` |

---

## Discrete planning

| API | Failure | Behavior | Tests |
|-----|---------|----------|-------|
| `AStar(grid, grid_type=...)` | unknown `grid_type` | Raise `ValueError` | `tests/planning/discrete/test_api.py` |
| `AStarPlanner.plan` / `plan_with_diagnostics` | start occupied | Sentinel `None` (diagnostics: `[], {}`) | `tests/planning/discrete/test_astar.py` |
| `AStarPlanner.plan` | no connecting path | Sentinel `None` | `tests/planning/discrete/test_astar.py` |
| `RouteRouter.plan` | start/goal beyond activation radius | Sentinel `None` | `tests/planning/discrete/` |
| `RouteRouter.plan` | no graph path between projections | Sentinel `None` | `tests/planning/discrete/` |
| `DStarLite.search` | any call | Raise `NotImplementedError` | `tests/planning/discrete/test_api.py` (xfail) |

---

## Continuous planning

| API | Failure | Behavior | Tests |
|-----|---------|----------|-------|
| `RRTPlanner` / `SSTPlanner` ctor | empty `bounds` | Raise `ValueError` | `tests/planning/continuous/` |
| `RRTPlanner` / `SSTPlanner` ctor | non-positive `step_size` | Raise `ValueError` | `tests/planning/continuous/` |
| `RRTPlanner.plan` / `SSTPlanner.plan` | occupied start or goal | Sentinel `None` | `tests/planning/continuous/` |
| `RRTPlanner.plan` / `SSTPlanner.plan` | no solution within sample budget | Sentinel `None` | `tests/planning/continuous/` |

---

## Control / guidance

| API | Failure | Behavior | Tests |
|-----|---------|----------|-------|
| `MPCController(...)` | construction | Warn `DeprecationWarning` | control MPC tests |
| `DubinsPathFollowingMPC` / `JointSpaceMPC` | CasADi not installed | Raise `ImportError` (optional extra `arco[mpc]`) | MPC tests / smoke with `[mpc]` |
| `BSplineInterpolator.interpolate` | any path | Returns input unchanged (stub; not a failure) | `tests/guidance/` |

---

## Acceptance criteria

1. Invalid constructor parameters that the API can detect raise `ValueError`
   (or a documented specific exception).
2. Unsatisfiable planning queries return `None` rather than inventing a path
   through occupied space.
3. Each row above has at least one automated test, or is explicitly marked
   as intentional stub / optional-extra behavior.
