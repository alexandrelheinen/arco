# Planning Layer Overview

The planning layer in ARCO provides algorithms for finding feasible paths through a spatial representation of the environment. It includes both graph search and sampling-based methods.

## Implemented Algorithms

### Discrete Planning
- **A\***: Grid and graph-based optimal path search (see [planning_astar.md](planning_astar.md))
- **Route Planning**: A* integration for road networks with waypoint smoothing

### Continuous Planning
- **RRT\***: Asymptotically optimal sampling-based planner (see [planning_rrt.md](planning_rrt.md))
- **SST**: Stable Sparse Trees for kinodynamic planning (see [planning_sst.md](planning_sst.md))

## Stub Implementations (Won't Do)
- **D\* Lite**: Dynamic replanning stub exists but full implementation is not planned (see [planning_dstar.md](planning_dstar.md))

## Directory Structure
```
src/arco/planning/
├── __init__.py
├── discrete/
│   ├── __init__.py
│   ├── base.py          ← DiscretePlanner abstract base
│   ├── astar.py         ← A* planner implementation
│   ├── dstar.py         ← D* Lite stub
│   ├── route.py         ← Route planning (A* for road networks)
│   └── api.py           ← Public API wrappers (AStar, DStarLite)
└── continuous/
    ├── __init__.py
    ├── base.py          ← ContinuousPlanner abstract base
    ├── rrt.py           ← RRT* planner implementation
    └── sst.py           ← SST planner implementation
```

## References
- See [README.md](../README.md) for global references.

---

*This document reflects the current state of the planning layer.*
