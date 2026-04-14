# ARCO Project Roadmap

## Core Layers
- ✅ **Mapping**: Spatial representation (graphs, grids, occupancy)
- ✅ **Planning**: Path search and sampling (see [PLANNING.md](PLANNING.md))
- ✅ **Guidance**: Path tracking and control (see [GUIDANCE.md](GUIDANCE.md))

## Completed Features
- [x] Project structure and dev setup
- [x] A* implementation and comprehensive tests
- [x] Grid structures (Manhattan, Euclidean)
- [x] Graph hierarchy (Graph → WeightedGraph → CartesianGraph → RoadGraph)
- [x] Route planning with A* integration
- [x] RRT* implementation (asymptotically optimal sampling-based planner)
- [x] SST implementation (Stable Sparse Trees)
- [x] Occupancy structures (abstract interface, KDTree implementation)
- [x] Guidance layer: Pure Pursuit, PID, MPC controllers
- [x] B-spline interpolation for path smoothing
- [x] Dubins primitive for kinematic constraints
- [x] Continuous integration and code coverage
- [x] Visualization tools and examples

## Current Status (Won't Do)
- D* Lite: Stub exists but full implementation is not planned
  - API wrapper exists in `planning.discrete.api.DStarLite`
  - Incremental replanning is handled by route planning instead

## Feature Milestones

### Horse Auto-Follow System
End-to-end integration demo for AC-style autonomous navigation (see [horse_auto_follow.md](horse_auto_follow.md))

**Phase 1: Static Path & Basic Steering** ✅ COMPLETE
- [x] Road graph extraction (mapping layer)
- [x] A* integration for pathfinding
- [x] Path smoothing (B-splines)
- [x] Pure Pursuit controller implementation
- [x] Route planning benchmarks and acceptance criteria

**Phase 2: Dynamic Tracking & Polish** (Future Work)
- [ ] Dynamic player tracking with replanning
- [ ] Local obstacle avoidance
- [ ] Behavioral polish and context-awareness
- [ ] Performance optimization

## Longer-Term Ideas
- ROS 2 integration
- More advanced kinodynamic planning variants
- Additional motion primitives
- Real-time visualization dashboard

---

## IPC & Telemetry Middleware

**Status**: Deferred — open design task.

### Current approach

The planner telemetry system (`arco.planning.continuous.telemetry`) uses a
JSON temp-file written by planners and polled by the loading screen.  This is
a functional but limited stopgap.

### Required architecture

Each pipeline step should run in its own OS process.  Steps communicate only
through well-defined message channels — see [docs/PIPELINE.md](PIPELINE.md)
for the full sequential pipeline description.

A pub/sub middleware would allow any module to publish its internal state
(iterations, cost, distance-to-goal) and any observer (loading screen, remote
dashboard, test harness) to subscribe without tight coupling.

### Technology evaluation task

**TODO**: Evaluate and select the IPC/telemetry technology.  Candidates:

| Technology | Pros | Cons |
|-----------|------|------|
| Redis pub/sub | Simple, fast, language-agnostic | External dependency, requires server |
| ZeroMQ (pyzmq) | No server, low latency, flexible | More complex API |
| MQTT (paho) | IoT-standard, broker-based | Needs broker, higher latency |
| DDS / ROS2 | Industry standard for robotics | Heavy dependency |
| Shared memory (multiprocessing) | Fast, no network | Python-only, no persistence |

Decision criteria: latency < 10 ms, no external server for local runs,
Python-first, extensible to remote monitoring.

Once selected, replace `arco.planning.continuous.telemetry` with a proper
publisher and update all consumers.

---

*This roadmap reflects the current state of the project as of the most recent documentation review.*
