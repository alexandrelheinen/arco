# Planning Layer Overview

The planning layer in ARCO provides algorithms for finding feasible paths through a spatial representation of the environment. It includes both graph search and sampling-based methods.

## Implemented Algorithms

### Discrete Planning
- **A\***: Grid and graph-based optimal path search (see [planning_astar.md](planning_astar.md))
- **Route Planning**: A* integration for road networks with waypoint smoothing

### Continuous Planning
- **RRT\***: Asymptotically optimal sampling-based planner (see [planning_rrt.md](planning_rrt.md))
- **SST**: Stable Sparse Trees for kinodynamic planning (see [planning_sst.md](planning_sst.md))
- **TrajectoryOptimizer**: Two-stage trajectory refinement for time-optimal execution (see [planning_optimizer.md](planning_optimizer.md))

## Stub Implementations (Won't Do)
- **D\* Lite**: Stub (`DStarLite` in `planning.discrete.api`) raises
  `NotImplementedError`. Full implementation is not planned — see
  [ROADMAP.md](ROADMAP.md).

## Directory Structure
```
src/arco/planning/
├── __init__.py
├── cost.py              ← PlannerCost (default distance + heuristic)
├── discrete/
│   ├── __init__.py
│   ├── base.py          ← DiscretePlanner (inherits PlannerCost)
│   ├── astar.py         ← A* planner implementation
│   ├── dstar.py         ← D* Lite stub
│   ├── route.py         ← Route planning (A* for road networks)
│   └── api.py           ← Public API wrappers (AStar, DStarLite)
└── continuous/
    ├── __init__.py
    ├── base.py          ← ContinuousPlanner (inherits PlannerCost)
    ├── rrt.py           ← RRT* planner implementation
    ├── sst.py           ← SST planner implementation
    └── optimizer.py     ← TrajectoryOptimizer (two-stage refinement)
```

## Cost Functions

A*, RRT*, and SST share :class:`~arco.planning.cost.PlannerCost` through
their discrete/continuous bases.  Override `distance` and/or `heuristic`
on a planner subclass to customize costs without rewriting search loops.

| Method | Default (`PlannerCost`) | Discrete override | Continuous override |
|---|---|---|---|
| `distance(a, b)` | Euclidean | `graph.distance` | step-size-normalized Euclidean |
| `heuristic(a, b)` | same as `distance` | `graph.heuristic` or `distance` | same as continuous `distance` |

A* also accepts an optional `heuristic=` callable in its constructor.
The trajectory optimizer keeps its own composite cost (weights only).

## Extension Points

Structural contracts live in `arco.protocols`.  Library defaults always
preserve historical behavior; optional overrides are opt-in.

| Hook | Where | Default | Override |
|---|---|---|---|
| `distance` / `heuristic` | `PlannerCost` (A*/RRT*/SST) | Euclidean / graph / step-normalized | Subclass or inject `cost=` |
| `heuristic=` callable | `AStarPlanner` ctor | `graph.heuristic` or `distance` | Pass callable |
| `sampler=` / `steerer=` / `segment_free=` | `RRTPlanner`, `SSTPlanner` | Uniform AABB, straight-line step, point occupancy samples | Callables / protocols |
| `cost=` | Continuous planners | Planner's own `PlannerCost` methods | `PlannerCost` instance |
| `planner=` | `RouteRouter` | Internal `AStarPlanner` | Any discrete planner with `plan` |
| `simplify_path` / `prefer_straight` | `AStarPlanner` | `True` / `True` | Disable via ctor flags |
| `publisher=` / `seed=` | Continuous planners | File telemetry / unseeded RNG | Custom sink / seed |
| `steer=` | `TrajectoryPruner.prune` | Occupancy segment check | Feasibility callable |
| `feasibility=` / `inverse_kinematics=` | `TrajectoryOptimizer.optimize` | None (geometry-only) | Callables |
| `cost_terms=` | `TrajectoryOptimizer` | Five default weighted terms | Replace/append terms |

### Guidance primitives and RRT-family planners

`:class:`~arco.guidance.primitive.base.ExplorationPrimitive`` (and
`DubinsPrimitive`) define a steerer ABC intended for kinodynamic RRT/SST.
They are **not** auto-wired into `RRTPlanner` / `SSTPlanner`.  Default
steering remains geometric straight-line extension.

To use a primitive, wrap it as a `steerer=` callable::

    primitive = DubinsPrimitive(...)
    planner = RRTPlanner(
        occ,
        bounds=bounds,
        steerer=lambda a, b: primitive.steer(a, b)[-1],
    )

Until a steerer is injected, treating RRT/SST as kinodynamic is incorrect.

## References
- See [README.md](../README.md) for global references.
- See `arco.protocols` for structural typing contracts.

---

*This document reflects the current state of the planning layer.*
