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

### Entity model (`arco.tools.entity`)
- Canonical typed hierarchy for all physical entities in an ARCO scene
- `Entity` ABC → `Agent` (`DubinsAgent`, `CartesianAgent`), `Link`, `Joint`
  (`RevoluteJoint`, `PrismaticJoint`), `EndEffector`, `Object`
- Two geometry descriptors: `BoxGeometry` (rectangle/cuboid) and `SphereGeometry`
  (circle/sphere); both JSON-serialisable
- `KinematicChain` — assembles Links, Joints, and an EndEffector into a
  manipulator description
- Formats evaluated and rationale documented in [entity_formats.md](entity_formats.md)

See [ENTITY_MODEL.md](ENTITY_MODEL.md) for details.

### Viewer (`arco.tools.viewer`)
- Unified rendering engine shared by all examples and the static image mode of `arcosim`
- `SceneSnapshot` — JSON-serialisable snapshot of a single planning result
  (obstacles, tree, found path, pruned path, trajectory, executed trace, metrics)
- `FrameRenderer` — renders any `SceneSnapshot` onto a matplotlib axes (2-D or 3-D)
- `StandardLayout` — composes workspace and C-space panels into a publication-ready figure
- `draw_grid`, `draw_graph`, `draw_road` — lower-level layer helpers
- Algorithm code writes a `SceneSnapshot`; the viewer reads it — no
  algorithm-specific rendering code

### Middleware and pipeline (`arco.middleware`, `arco.pipeline`)
- `Bus` ABC + `InMemoryBus` — typed, thread-safe, bounded message bus
- `BusPublisher` / `BusSubscriber` mixins for pipeline nodes and frontends
- `MappingFrame`, `PlanFrame`, `GuidanceFrame` — typed arc dataclasses
- `PipelineNode` ABC — lifecycle-managed thread that publishes frames to the bus
- `PipelineRunner` — wires nodes to a shared bus and manages start/stop

---

## What is not planned

| Feature | Decision |
|---------|----------|
| **D\* Lite** | Stub exists (`DStarLite` in `planning.discrete.api`), full implementation not planned. Route planning with A* covers the main use case; incremental replanning is not required. See [planning_dstar.md](planning_dstar.md). |
| **IPC / pub-sub middleware** | Full IPC across processes is not planned. The in-process `arco.middleware` bus (implemented) is sufficient for the single-process pipeline. |
