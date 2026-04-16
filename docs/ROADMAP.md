# ARCO Project Roadmap

## What is in the library now

### Mapping layer
- Grid structures: `ManhattanGrid` (4-connected) and `EuclideanGrid` (8-connected)
- Graph hierarchy: `Graph` → `WeightedGraph` → `CartesianGraph` → `RoadGraph`
- Road network loader: `load_road_graph()` reads JSON with node positions and edge waypoints
- Occupancy interface (`Occupancy` ABC) and `KDTreeOccupancy` implementation

See [MAPPING.md](MAPPING.md) for details.

### Planning layer
- A* planner (`AStarPlanner` / `AStar` wrapper) for grids and graphs
- Route planning (`RouteRouter`) — A* on road networks with nearest-node projection
- RRT* planner (`RRTPlanner`) — asymptotically optimal sampling-based planning
- SST planner (`SSTPlanner`) — stable sparse trees for memory-efficient planning
- Trajectory optimizer (`TrajectoryOptimizer`) — two-stage path refinement
- Trajectory pruner (`TrajectoryPruner`) — removes redundant waypoints

See [PLANNING.md](PLANNING.md) for details.

### Guidance layer
- Controllers: `PIDController`, `PurePursuitController`, `MPCController`, `TrackingLoop`
- Interpolation: `BSplineInterpolator` for smooth trajectory generation
- Primitives: `DubinsPrimitive` for kinematic steering constraints
- Vehicle models: `DubinsVehicle`

See [GUIDANCE.md](GUIDANCE.md) for details.

### Tools
- `arcosim` — unified CLI for real-time simulation and static image generation
  (`--image` mode) from scenario YAML files (pygame/PyOpenGL + matplotlib)
- Built-in scenarios: `astar`, `city`, `occ`, `ppp`, `rr`, `rrp`, `vehicle`

See [VISUALIZATION.md](VISUALIZATION.md) for details.

---

## What is not planned

| Feature | Decision |
|---------|----------|
| **D\* Lite** | Stub exists (`DStarLite` in `planning.discrete.api`), full implementation not planned. Route planning with A* covers the main use case; incremental replanning is not required. See [planning_dstar.md](planning_dstar.md). |
| **IPC / pub-sub middleware** | Not planned. Telemetry currently uses a JSON temp-file (`arco.planning.continuous.telemetry`), which is sufficient for the current single-process pipeline. |
