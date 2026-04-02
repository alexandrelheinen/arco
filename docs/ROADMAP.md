# ARCO Project Roadmap

## Core Layers
- **Mapping**: Spatial representation (TBD)
- **Planning**: Path search and sampling (see [PLANNING.md](PLANNING.md))
- **Guidance**: Path tracking and control (TBD)

## Near-Term Goals
- [x] Project structure and dev setup
- [x] A* implementation and test
- [x] D* stub and test
- [ ] D* Lite implementation
- [ ] RRT and RRT* stubs
- [ ] Guidance layer: pure pursuit, PID
- [ ] Mapping layer: occupancy grid

## Feature Milestones

### Horse Auto-Follow System
End-to-end integration demo for AC-style autonomous navigation (see [horse_auto_follow.md](horse_auto_follow.md))

**Phase 1: Static Path & Basic Steering**
- [ ] Road graph extraction (mapping layer)
- [ ] A* integration for pathfinding
- [ ] Path smoothing (B-splines/Catmull-Rom)
- [ ] Pure Pursuit controller implementation
- [ ] Physics integration and tuning

**Phase 2: Dynamic Tracking & Polish**
- [ ] Dynamic player tracking with replanning
- [ ] D* Lite integration (optional optimization)
- [ ] Local obstacle avoidance
- [ ] Behavioral polish and context-awareness
- [ ] Performance optimization

## Longer-Term
- ROS 2 integration
- Continuous integration and code coverage
- Visualization tools and example notebooks
- More advanced planners (SST, kinodynamic RRT*)

---

*This roadmap will be updated as the project evolves.*
