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

*This roadmap reflects the current state of the project as of the most recent documentation review.*
