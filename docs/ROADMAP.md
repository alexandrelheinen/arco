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

### Guidance / control layer
- Controllers: `PIDController`, `PurePursuitController`, `TrackingLoop`
- Path-following MPC (shipped): `DubinsPathFollowingMPC`, `MPCTrackingLoop`,
  `ReferencePath` under `arco.control.mpc` (optional `arco[mpc]` / CasADi)
- Joint-space MPC (shipped): `JointSpaceMPC` drop-in for `JointSpaceTracker`
  (PPP / RRP carrot tracking; select via `simulator.tracker: mpc`)
- Deprecated stub: scalar `MPCController` (use `DubinsPathFollowingMPC`)
- Interpolation: `BSplineInterpolator` for smooth trajectory generation
- Primitives: `DubinsPrimitive` for kinematic steering constraints
- Vehicle models: `DubinsVehicle`

See [GUIDANCE.md](GUIDANCE.md) for details.

### Follow-ups
- Optional acados backend after the CasADi formulation is stable
- Full contouring joint-space MPC (path progress state) beyond carrot NMPC

### Tools
- `arcosim` — unified CLI for real-time simulation and static image generation
  (`--image` mode) from scenario YAML files (pygame/PyOpenGL + matplotlib)
- Built-in scenarios: `astar`, `city`, `occ`, `ppp`, `rr`, `rrp`, `vehicle`

See [VISUALIZATION.md](VISUALIZATION.md) for details.

### Entity model (`arco.simulator.entity`)
- Canonical typed hierarchy for physical entities in an ARCO scene
- `Entity` ABC → `Agent` (`DubinsAgent`, `CartesianAgent`), `Link`, `Joint`
  (`RevoluteJoint`, `PrismaticJoint`), `EndEffector`, `Object`
- Geometry descriptors: `BoxGeometry`, `SphereGeometry` (JSON-serialisable)
- `KinematicChain` — Links, Joints, and an EndEffector
- Format research note: [entity_formats.md](entity_formats.md)

See [ENTITY_MODEL.md](ENTITY_MODEL.md) for details.

### Viewer (`arco.simulator.viewer`)
- `SceneSnapshot` — JSON-serialisable planning result snapshot
- `FrameRenderer` — renders a `SceneSnapshot` onto a matplotlib axes
- `StandardLayout` — workspace / C-space figure composition
- `draw_grid`, `draw_graph`, `draw_road` — layer helpers
- Not yet the exclusive render path for all `arcosim` mains (OpenGL/pygame
  paths still dominate)

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
| **D\* Lite** | Stub exists (`DStarLite` in `planning.discrete.api`), full implementation not planned. Route planning with A* covers the main use case; incremental replanning is not required. |
| **IPC / pub-sub middleware** | Full IPC across processes is not planned. The in-process `arco.middleware` bus (implemented) is sufficient for the single-process pipeline. |
