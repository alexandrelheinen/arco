# D* Lite Path Planning

D* Lite (Dynamic A*) is an incremental heuristic search algorithm for dynamic environments, allowing efficient replanning as the environment changes. It is widely used in robotics for navigation in partially-known or changing maps.

## Key Reference
- Stentz, A. (1994). Optimal and Efficient Path Planning for Partially-Known Environments. Proceedings of the IEEE International Conference on Robotics and Automation.
- Koenig, S. & Likhachev, M. (2002). D* Lite. AAAI Conference on Artificial Intelligence.

## Algorithm Overview
- Maintains a cost-to-goal map that can be efficiently updated as obstacles are discovered or removed.
- Supports dynamic replanning without recomputing the entire path from scratch.
- D* Lite is a simplified variant that is easier to understand and implement.

## Current Status

⏸️ **Stub Only - Not Planned for Full Implementation**

A stub implementation exists in `src/arco/planning/discrete/dstar.py` with a public API wrapper in `src/arco/planning/discrete/api.py`, but full D* Lite functionality is **not planned** for implementation.

### Why Not Implemented?
- Route planning with A* already handles the main use case for road networks
- For dynamic replanning, simple re-invocation of A* is sufficient for most scenarios
- D* Lite adds complexity that is not justified by current project requirements

### What Exists
- [x] Stub class `DStarPlanner` in `planning.discrete.dstar`
- [x] Public API wrapper `DStarLite` in `planning.discrete.api`
- [x] Raises `NotImplementedError` when `plan()` is called

If dynamic replanning is needed in the future, the stub provides a clear extension point.

## Resources
- [Planning Algorithms, Ch. 10.2](http://planning.cs.uiuc.edu/node198.html)
- [Wikipedia: D* Lite](https://en.wikipedia.org/wiki/D*_Lite)
- [Original D* Lite Paper](http://idm-lab.org/bib/abstracts/papers/aaai02b.pdf)

---

*This document reflects the current decision not to implement D* Lite.*
