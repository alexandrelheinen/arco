# Planning Layer Overview

The planning layer in ARCO provides algorithms for finding feasible paths through a spatial representation of the environment. It includes both graph search and sampling-based methods.

## Implemented Algorithms
- **A\***: Grid-based, optimal path search (see [planning_astar.md](planning_astar.md))
- **D\***: Dynamic replanning for changing environments (see [planning_dstar.md](planning_dstar.md))

## Roadmap
- [x] A* (grid-based)
- [x] D* (stub)
- [ ] D* Lite (full implementation)
- [ ] RRT (Rapidly-exploring Random Tree)
- [ ] RRT*
- [ ] SST (Stable Sparse RRT)

## Directory Structure
```
src/arco/planning/
    astar.py
    dstar.py
    ...
```

## References
- See [README.md](../README.md) for global references.

---

*This document will be updated as new planners are added.*
