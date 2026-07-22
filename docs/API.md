# ARCO Public API Reference

Authoritative list of classes and functions library users should import.
Anything not listed here is an implementation detail and may change without
notice.

For layer narrative, see [MAPPING.md](MAPPING.md), [PLANNING.md](PLANNING.md),
and [GUIDANCE.md](GUIDANCE.md).

---

## Import conventions

Prefer layer packages (`arco.mapping`, `arco.planning`, `arco.control`, …).
Each package `__init__.py` is re-export-only.

```python
from arco.mapping import (
    Grid, ManhattanGrid, EuclideanGrid,
    Graph, WeightedGraph, CartesianGraph, RoadGraph,
    load_road_graph,
    Occupancy, KDTreeOccupancy,
)
from arco.planning import (
    AStar, AStarPlanner, DiscretePlanner,
    RouteResult, RouteRouter,
    ContinuousPlanner, RRTPlanner, SSTPlanner,
    TrajectoryOptimizer, TrajectoryResult, TrajectoryPruner,
    PlanningPipeline, PipelineResult,
)
from arco.control import (
    Controller, PIDController, PurePursuitController, TrackingLoop,
    DubinsPathFollowingMPC, MPCTrackingLoop, ReferencePath,
    JointSpaceMPC, JointSpaceTracker, ActuatorArray,
    RigidBody, CircleBody, SquareBody,
)
from arco.guidance import (
    BSplineInterpolator, Interpolator,
    DubinsPrimitive, ExplorationPrimitive,
    DubinsVehicle,
)
from arco.kinematics import RRRobot, RRPRobot
from arco.middleware import Bus, InMemoryBus, BusPublisher, BusSubscriber
from arco.middleware import MappingFrame, PlanFrame, GuidanceFrame
from arco.pipeline import PipelineNode, PipelineRunner
```

---

## Public API pattern

| Rule | Detail |
|------|--------|
| One main class per module | Matching [guidelines.md](guidelines.md) §1 |
| Import from layer package | `from arco.planning import RRTPlanner`, not deep private paths |
| `__all__` is the contract | Every public package declares `__all__` |
| Stable return sentinels | Planners return `None` when no path exists |
| Explicit constructor errors | Invalid parameters raise `ValueError` (or documented exceptions) |
| Deprecations warn once | Deprecated symbols emit `DeprecationWarning` and point to the replacement |

---

## Mapping (`arco.mapping`)

| Symbol | Module | Role |
|--------|--------|------|
| `Grid` | `mapping/grid/base.py` | Abstract N-D grid (`neighbors`, `distance`, `is_occupied`, …) |
| `ManhattanGrid` | `mapping/grid/manhattan.py` | 4-connected, L₁ |
| `EuclideanGrid` | `mapping/grid/euclidean.py` | 8-connected, L₂ |
| `Graph` | `mapping/graph/base.py` | Topology base |
| `WeightedGraph` | `mapping/graph/weighted.py` | Weighted edges + A* interface |
| `CartesianGraph` | `mapping/graph/cartesian.py` | Node positions + nearest/project helpers |
| `RoadGraph` | `mapping/graph/road.py` | Per-edge waypoints (`full_edge_geometry`) |
| `load_road_graph` | `mapping/graph/loader.py` | Load road JSON → `RoadGraph` |
| `Occupancy` | `mapping/occupancy.py` | Continuous obstacle ABC |
| `KDTreeOccupancy` | `mapping/kdtree.py` | Point-cloud occupancy (`clearance` radius) |

```python
from arco.mapping import ManhattanGrid, KDTreeOccupancy
import numpy as np

g = ManhattanGrid(shape=(100, 100))
g.data[40:60, 40:60] = 1

occ = KDTreeOccupancy(np.array([[5.0, 5.0]]), clearance=0.5)
assert occ.is_occupied(np.array([5.1, 5.1]))
```

`KDTreeOccupancy` raises `ValueError` if `points` is empty or `clearance <= 0`.

---

## Planning (`arco.planning`)

### Discrete

| Symbol | Module | Role |
|--------|--------|------|
| `DiscretePlanner` | `planning/discrete/base.py` | ABC: `plan(start, goal)` |
| `AStarPlanner` | `planning/discrete/astar.py` | A* on any graph/grid; `plan`, `plan_with_diagnostics`. Occupied start → `None`. |
| `AStar` | `planning/discrete/api.py` | Numpy-grid wrapper; `search(start, goal)`. Raises `ValueError` for unknown `grid_type`. |
| `RouteRouter` | `planning/discrete/route.py` | Project poses → A* on Cartesian/Road graphs |
| `RouteResult` | `planning/discrete/route.py` | Named result of `RouteRouter.plan` |
| `DStarLite` / `DStarPlanner` | stub | Raises `NotImplementedError` (not planned) |

```python
from arco.planning import AStar, AStarPlanner
from arco.mapping import EuclideanGrid
import numpy as np

grid = np.zeros((20, 20), dtype=np.uint8)
path = AStar(grid, grid_type="euclidean").search((0, 0), (19, 19))

g = EuclideanGrid(shape=(20, 20))
g.data = grid
path, expanded, parents = AStarPlanner(g).plan_with_diagnostics((0, 0), (19, 19))
```

### Continuous

| Symbol | Module | Role |
|--------|--------|------|
| `ContinuousPlanner` | `planning/continuous/base.py` | ABC: `plan(start, goal)` |
| `RRTPlanner` | `planning/continuous/rrt.py` | RRT*; `plan`, `get_tree` |
| `SSTPlanner` | `planning/continuous/sst.py` | SST; `plan`, `get_tree` |
| `TrajectoryPruner` | `planning/continuous/pruner.py` | Minimum-hop shortcut pruner |
| `TrajectoryOptimizer` | `planning/continuous/optimizer.py` | Two-stage scipy refinement |
| `TrajectoryResult` | `planning/continuous/optimizer.py` | Optimizer output dataclass |
| `PlanningPipeline` | `planning/pipeline.py` | planner → pruner → optimizer |
| `PipelineResult` | `planning/pipeline.py` | Stage outputs + timings |

`RRTPlanner` / `SSTPlanner` raise `ValueError` for empty `bounds` or
non-positive `step_size`. Occupied start/goal yields `None` (no exception).

---

## Control (`arco.control`)

| Symbol | Role |
|--------|------|
| `Controller` | ABC: `control(state, reference)` |
| `PIDController` | Classic PID |
| `PurePursuitController` | Geometric look-ahead tracker |
| `TrackingLoop` | Pure Pursuit integration (+ optional APF) |
| `DubinsPathFollowingMPC` | CasADi contouring NMPC (`arco[mpc]`) |
| `MPCTrackingLoop` | Metrics loop for path-following MPC |
| `ReferencePath` / `PathFollowingMPCConfig` / `DubinsVehicleLimits` | MPC inputs |
| `JointSpaceMPC` / `JointSpaceMPCConfig` | N-DOF carrot NMPC |
| `JointSpaceTracker` / `ActuatorArray` | Joint tracking without CasADi |
| `RigidBody` / `CircleBody` / `SquareBody` | Object-centric geometry |
| `MPCController` | **Deprecated** scalar stub |
| `MPCTracker` / `MPCStepResult` | MPC tracker ABC / step result |

---

## Guidance (`arco.guidance`)

| Symbol | Role |
|--------|------|
| `Interpolator` / `BSplineInterpolator` | Path smoothing (`interpolate`; B-spline is a stub) |
| `ExplorationPrimitive` / `DubinsPrimitive` | Steering primitives |
| `DubinsVehicle` | Car-like kinematic model |

`arco.guidance` also re-exports `Controller`, `PIDController`,
`PurePursuitController`, `TrackingLoop`, and deprecated `MPCController` from
`arco.control` for convenience. Prefer importing controllers from
`arco.control`.

---

## Kinematics (`arco.kinematics`)

| Symbol | Role |
|--------|------|
| `RRRobot` | 2-DOF RR arm FK/IK |
| `RRPRobot` | RRP arm FK/IK |

Used by manipulator scenarios (PPP / RRP) and joint-space tracking.

---

## Middleware and pipeline

| Symbol | Package | Role |
|--------|---------|------|
| `Bus` / `InMemoryBus` | `arco.middleware` | Typed in-process bus |
| `BusPublisher` / `BusSubscriber` | `arco.middleware` | Publish/subscribe mixins |
| `MappingFrame` / `PlanFrame` / `GuidanceFrame` | `arco.middleware` | Typed stage frames |
| `PipelineNode` / `PipelineRunner` | `arco.pipeline` | Threaded stage lifecycle |

These are public library APIs. The current `arcosim` runtime does not yet
wire them as its primary execution path.

---

## What exists but is not the library interface

Identified for cleanup / non-use by library consumers:

| Symbol / area | Why it is irrelevant to the public library interface |
|---------------|------------------------------------------------------|
| `arco.planning.continuous.telemetry` | Internal loading-screen IPC side-channel |
| `arco.config.load_config` | Shared YAML loader for package configs, not a feature API |
| `arco.simulator.*` | Application layer (`arcosim` CLI); not a stable library API |
| `MPCController` | Deprecated stub; use `DubinsPathFollowingMPC` |
| `DStarLite` / `DStarPlanner` | Intentional non-feature stubs (kept only as API placeholders) |
| `BSplineInterpolator` body | Stub pass-through; symbol kept, algorithm not shipped |

Removed in the cleanup (were listed as non-interface above):

- `arco.guidance.control.*` deprecated shims
- `arco.mapping.graph.OrientedGraph` unused empty subclass
- Documented-but-missing `load_map_config`

---

## Related docs

- [FAILURE_MODES.md](FAILURE_MODES.md) — constructor and planning failure contracts
- [ALGORITHMS.md](ALGORITHMS.md) — core computational blocks
- [ROADMAP.md](ROADMAP.md) — shipped vs won't-do features
- Layer overviews: [MAPPING.md](MAPPING.md), [PLANNING.md](PLANNING.md), [GUIDANCE.md](GUIDANCE.md)
