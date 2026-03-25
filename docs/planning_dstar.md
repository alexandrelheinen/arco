# D* (Dynamic A*) Path Planning

D* (Dynamic A*) is an incremental heuristic search algorithm for dynamic environments, allowing efficient replanning as the environment changes. It is widely used in robotics for navigation in partially-known or changing maps.

## Key Reference
- Stentz, A. (1994). Optimal and Efficient Path Planning for Partially-Known Environments. Proceedings of the IEEE International Conference on Robotics and Automation.

## Algorithm Overview
- Maintains a cost-to-goal map that can be efficiently updated as obstacles are discovered or removed.
- Supports dynamic replanning without recomputing the entire path from scratch.
- D* Lite is a popular simplified variant.

## Implementation Plan
- [x] Stub class and test
- [ ] Implement D* Lite core algorithm
- [ ] Add dynamic obstacle update interface
- [ ] Add visualization and example notebook

## Resources
- [Planning Algorithms, Ch. 10.2](http://planning.cs.uiuc.edu/node198.html)
- [Wikipedia: D* Lite](https://en.wikipedia.org/wiki/D*_Lite)

---

*This document will be updated as the implementation progresses.*
