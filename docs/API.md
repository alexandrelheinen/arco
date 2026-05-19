# ARCO Public API Reference

This document is the authoritative reference for the ARCO public API — the
classes, functions, and constants that library users are expected to import and
call directly.  Anything not listed here is an implementation detail and may
change without notice.

---

## Import conventions

```python
# Mapping layer
from arco.mapping import (
    Grid, ManhattanGrid, EuclideanGrid,
    Graph, WeightedGraph, CartesianGraph, RoadGraph,
    load_road_graph,
    Occupancy, KDTreeOccupancy,
)

# Planning layer
from arco.planning import (
    AStar, AStarPlanner, DiscretePlanner,
    DStarLite,                          # stub — raises NotImplementedError
    RouteResult, RouteRouter,
    ContinuousPlanner,
    RRTPlanner, SSTPlanner,
    TrajectoryOptimizer, TrajectoryResult,
    TrajectoryPruner,
    PlanningPipeline, PipelineResult,
)

# Control layer
from arco.control import (
    Controller,
    PIDController, PurePursuitController, MPCController,
    TrackingLoop,
    ActuatorArray, JointSpaceTracker,
    RigidBody, CircleBody, SquareBody,
)

# Guidance layer (interpolation + primitives + vehicle models)
from arco.guidance import (
    BSplineInterpolator, Interpolator,
    DubinsPrimitive, ExplorationPrimitive,
    DubinsVehicle,
)

# Middleware and pipeline (in-process message bus)
from arco.middleware import Bus, BusPublisher, BusSubscriber
from arco.middleware.types import MappingFrame, PlanFrame, GuidanceFrame
from arco.pipeline import PipelineNode, PipelineRunner
```

---

## Mapping layer (`arco.mapping`)

### Grids

| Class | File | Description |
|-------|------|-------------|
| `Grid` | `mapping/grid/base.py` | Abstract N-dimensional grid base. Attributes: `shape`, `data`, `cell_size`, `physical_size`. Methods: `neighbors`, `distance`, `heuristic`, `position`, `is_occupied`, `set_occupied`, `set_free`. |
| `ManhattanGrid` | `mapping/grid/manhattan.py` | 4-connected grid (cardinal directions only). Distance metric: L₁. |
| `EuclideanGrid` | `mapping/grid/euclidean.py` | 8-connected grid (cardinal + diagonal). Distance metric: L₂. |

#### Grid construction

```python
from arco.mapping import ManhattanGrid, EuclideanGrid
import numpy as np

# By cell shape
g = ManhattanGrid(shape=(100, 100))
g.data[40:60, 40:60] = 1           # mark obstacle region

# By physical size (meters)
g = EuclideanGrid(physical_size=(50.0, 50.0), cell_size=0.5)
```

---

### Graph hierarchy

| Class | File | Description |
|-------|------|-------------|
| `Graph` | `mapping/graph/base.py` | Pure topology base (nodes + edges, no weights). |
| `WeightedGraph` | `mapping/graph/weighted.py` | Adds numeric edge weights. Methods: `add_node`, `add_edge`, `neighbors`, `distance`, `find_nearest_node`, `project_to_nearest_edge`. |
| `CartesianGraph` | `mapping/graph/cartesian.py` | Adds N-D node positions (numpy arrays). Edge weights default to Euclidean distance. Methods: `heuristic`, `find_nearest_node`, `project_to_nearest_edge`. |
| `RoadGraph` | `mapping/graph/road.py` | Adds per-edge intermediate waypoints for road geometry. Method: `full_edge_geometry(a, b)`. |
| `load_road_graph` | `mapping/graph/loader.py` | Load a road network from a JSON file. Returns `RoadGraph`. |

#### Graph usage

```python
from arco.mapping import CartesianGraph, RoadGraph, load_road_graph
import numpy as np

# Manual CartesianGraph
g = CartesianGraph()
g.add_node(0, 0.0, 0.0)
g.add_node(1, 3.0, 4.0)
g.add_edge(0, 1)            # weight = Euclidean distance (5.0)

# Load road network from JSON
road = load_road_graph("map/city.json")
```

---

### Occupancy maps

| Class | File | Description |
|-------|------|-------------|
| `Occupancy` | `mapping/occupancy.py` | Abstract interface for continuous obstacle maps. Abstract methods: `is_occupied(point)`, `nearest_obstacle(point)`. |
| `KDTreeOccupancy` | `mapping/kdtree.py` | KD-tree backed point-cloud obstacle map. Constructor: `KDTreeOccupancy(points, clearance)`. Additional methods: `query_distances(points)`, `dimension`, `points`. |

#### Occupancy usage

```python
from arco.mapping import KDTreeOccupancy
import numpy as np

obstacles = np.array([[5.0, 5.0], [5.0, 6.0], [6.0, 5.0]])
occ = KDTreeOccupancy(obstacles, clearance=0.5)

in_collision = occ.is_occupied(np.array([5.1, 5.1]))   # True
dist, pt = occ.nearest_obstacle(np.array([3.0, 3.0]))  # (2.8..., [5,5])
```

**Constructor raises:**
- `ValueError` if `points` is empty.
- `ValueError` if `clearance <= 0`.

---

## Planning layer (`arco.planning`)

### Discrete planners

| Class / function | File | Description |
|------------------|------|-------------|
| `DiscretePlanner` | `planning/discrete/base.py` | Abstract base. Method: `plan(start, goal) → list \| None`. |
| `AStarPlanner` | `planning/discrete/astar.py` | A* on any graph/grid. Constructor: `AStarPlanner(graph, heuristic=None)`. Methods: `plan`, `plan_with_diagnostics`. |
| `AStar` | `planning/discrete/api.py` | High-level wrapper. Constructor: `AStar(grid_array, grid_type='manhattan')`. Method: `search(start, goal)`. |
| `DStarLite` | `planning/discrete/api.py` | **Stub — raises `NotImplementedError`**. Present for API completeness. |
| `RouteRouter` | `planning/discrete/route.py` | A* on Cartesian graphs with position projection. Constructor: `RouteRouter(graph, activation_radius=None)`. Method: `plan(start_position, goal_position) → RouteResult \| None`. |
| `RouteResult` | `planning/discrete/route.py` | `NamedTuple` with fields: `path`, `start_node`, `goal_node`, `start_projection`, `goal_projection`, `start_distance`, `goal_distance`. |

#### A* usage

```python
from arco.planning import AStar, AStarPlanner
from arco.mapping import EuclideanGrid
import numpy as np

# High-level wrapper (grid array as input)
grid = np.zeros((20, 20), dtype=np.uint8)
grid[8:12, 5:15] = 1
planner = AStar(grid, grid_type='euclidean')
path = planner.search((0, 0), (19, 19))

# Low-level — with a Grid object
g = EuclideanGrid(shape=(20, 20))
g.data = grid
astar = AStarPlanner(g)
path, expanded, parent_map = astar.plan_with_diagnostics((0, 0), (19, 19))
```

---

### Continuous planners

| Class | File | Description |
|-------|------|-------------|
| `ContinuousPlanner` | `planning/continuous/base.py` | Abstract base. Method: `plan(start, goal) → list[np.ndarray] \| None`. |
| `RRTPlanner` | `planning/continuous/rrt.py` | Asymptotically-optimal RRT*. Constructor: `RRTPlanner(occupancy, bounds, ...)`. Methods: `plan`, `get_tree`. |
| `SSTPlanner` | `planning/continuous/sst.py` | Stable Sparse Trees. Constructor: `SSTPlanner(occupancy, bounds, ...)`. Methods: `plan`, `get_tree`. |
| `TrajectoryPruner` | `planning/continuous/pruner.py` | BFS-optimal path pruner. Constructor: `TrajectoryPruner(occupancy, step_size, ...)`. Method: `prune(path, steer=None)`. |
| `TrajectoryOptimizer` | `planning/continuous/optimizer.py` | Two-stage scipy path optimizer. Constructor: `TrajectoryOptimizer(occupancy, ...)`. Methods: `optimize(path, ...)`, `create_from_config`. |
| `TrajectoryResult` | `planning/continuous/optimizer.py` | Dataclass: `states`, `commands`, `durations`, `cost`, `is_feasible`, `optimizer_success`, `optimizer_status_code`, `optimizer_status_text`, `optimizer_iteration_count`. |

#### RRT* usage

```python
from arco.mapping import KDTreeOccupancy
from arco.planning import RRTPlanner
import numpy as np

occ = KDTreeOccupancy(obstacle_points, clearance=0.5)
planner = RRTPlanner(
    occupancy=occ,
    bounds=[(0, 50), (0, 50)],
    max_sample_count=2000,
    step_size=1.0,
    goal_tolerance=1.0,
    goal_bias=0.05,
    early_stop=True,
)
path = planner.plan(np.array([1.0, 1.0]), np.array([49.0, 49.0]))

# With tree export for visualization
nodes, parent, path = planner.get_tree(start, goal)
```

**Constructor raises:**
- `ValueError` if `bounds` is empty.
- `ValueError` if any `step_size` component is not positive.

---

### Planning pipeline

| Class | File | Description |
|-------|------|-------------|
| `PlanningPipeline` | `planning/pipeline.py` | Chains planner → pruner → optimizer. Methods: `run(start, goal)`, `run_from_path(path)`, `save_result`, `load_result`. |
| `PipelineResult` | `planning/pipeline.py` | Dataclass with stage outputs and timing: `raw_path`, `pruned_path`, `trajectory`, `durations`, `total_duration`, `planner_status`, `optimizer_status`, plus timing fields. |

```python
from arco.planning import PlanningPipeline, RRTPlanner, TrajectoryPruner
from arco.planning import TrajectoryOptimizer

pipeline = PlanningPipeline(
    planner=RRTPlanner(occ, bounds=bounds, step_size=step),
    pruner=TrajectoryPruner(occ, step_size=np.array([step, step])),
    optimizer=TrajectoryOptimizer.create_from_config(occ, cruise_speed=1.0),
)
result = pipeline.run(start, goal)
if result.trajectory:
    PlanningPipeline.save_result(result, "cache/run.npz")
```

---

## Control layer (`arco.control`)

| Class | File | Description |
|-------|------|-------------|
| `Controller` | `control/base.py` | Abstract controller base. Method: `control(state, reference) → float`. |
| `PIDController` | `control/pid.py` | PID feedback controller. Constructor: `PIDController(kp, ki, kd, dt, ...)`. Method: `control(state, reference)`. |
| `PurePursuitController` | `control/pure_pursuit.py` | Geometric look-ahead path tracker. Constructor: `PurePursuitController(lookahead_distance, wheelbase, ...)`. Methods: `control`, `set_path`. |
| `MPCController` | `control/mpc.py` | Model Predictive Controller. Constructor: `MPCController(model, horizon, ...)`. Method: `control`. |
| `TrackingLoop` | `control/tracking.py` | Wraps a controller with timing and state management. Method: `step(state)`. |
| `ActuatorArray` | `control/actuator.py` | Array of actuators with joint limits. |
| `JointSpaceTracker` | `control/joint_tracker.py` | Joint-space trajectory tracker. |
| `RigidBody` | `control/rigid_body/base.py` | Abstract rigid-body geometry for object-centric control. |
| `CircleBody` | `control/rigid_body/circle.py` | Circular rigid body. |
| `SquareBody` | `control/rigid_body/square.py` | Square/rectangular rigid body. |

---

## Guidance layer (`arco.guidance`)

### Interpolation

| Class | File | Description |
|-------|------|-------------|
| `Interpolator` | `guidance/interpolation/base.py` | Abstract interpolator base. Methods: `fit(waypoints)`, `evaluate(t)`. |
| `BSplineInterpolator` | `guidance/interpolation/bspline.py` | B-spline (C² continuous) curve through waypoints. Constructor: `BSplineInterpolator(degree=3)`. Method: `interpolate(path)` — currently a pass-through stub. |

```python
from arco.guidance import BSplineInterpolator
import numpy as np

interp = BSplineInterpolator(degree=3)
smooth_path = interp.interpolate(path)  # Returns smoothed list of waypoints
```

### Primitives

| Class | File | Description |
|-------|------|-------------|
| `ExplorationPrimitive` | `guidance/primitive/base.py` | Abstract exploration primitive. Method: `steer(from_state, to_state)`. |
| `DubinsPrimitive` | `guidance/primitive/dubins.py` | Dubins path steering for car-like vehicles. Constructor: `DubinsPrimitive(turning_radius)`. |

### Vehicle models

| Class | File | Description |
|-------|------|-------------|
| `DubinsVehicle` | `guidance/vehicle.py` | Car-like kinematic model (forward-only, fixed turn radius). Constructor: `DubinsVehicle(max_speed, ...)`. Methods: `step`, `inverse_kinematics`, `is_feasible`. |

---

## Middleware (`arco.middleware`)

| Class | File | Description |
|-------|------|-------------|
| `Bus` | `middleware/bus.py` | Abstract typed message bus. Methods: `publish(topic, msg)`, `subscribe(topic)`. |
| `BusPublisher` | `middleware/publisher.py` | Mixin for pipeline nodes that publish frames. |
| `BusSubscriber` | `middleware/subscriber.py` | Mixin for pipeline nodes that consume frames. |
| `MappingFrame` | `middleware/types/mapping_frame.py` | Typed dataclass for mapping stage output. |
| `PlanFrame` | `middleware/types/plan_frame.py` | Typed dataclass for planning stage output. |
| `GuidanceFrame` | `middleware/types/guidance_frame.py` | Typed dataclass for guidance stage output. |

---

## Pipeline nodes (`arco.pipeline`)

| Class | File | Description |
|-------|------|-------------|
| `PipelineNode` | `pipeline/node.py` | Abstract lifecycle-managed thread. Methods: `start`, `stop`, `run` (abstract). |
| `PipelineRunner` | `pipeline/runner.py` | Wires `PipelineNode` instances to a shared bus and manages lifecycle. |

---

## What is *not* public API

The following exist in the codebase but are **not** intended for direct user
consumption — they are implementation details, deprecated shims, or stubs:

| Module / symbol | Reason |
|-----------------|--------|
| `arco.guidance.control.*` | Deprecated re-export shim. Import from `arco.control` directly. |
| `arco.mapping.graph.oriented.OrientedGraph` | Defined but not used anywhere in the library or tools. Kept for completeness. |
| `arco.planning.discrete.dstar.DStarPlanner` | Internal stub class behind `DStarLite`. Use `DStarLite` from `arco.planning`. |
| `arco.planning.continuous.telemetry.*` | Internal telemetry side-channel for the loading screen; not for user use. |
| `arco.simulator.*` | Visualization tools (`arcosim` CLI). Internal to the tools layer; not a stable user API. |
| `arco.config.*` | Internal config loading utilities. Not a user API. |
| `arco.kinematics.*` | Robot arm kinematics models used by simulator scenes. Not a stable user API. |
