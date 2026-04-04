# RRT* Path Planning

RRT* (Rapidly-exploring Random Tree Star) is an asymptotically optimal sampling-based motion planning algorithm for continuous state spaces. It builds upon the original RRT algorithm by adding a rewiring step that progressively improves path quality as more samples are added.

## Key References
- LaValle, S. M. (1998). Rapidly-Exploring Random Trees: A New Tool for Path Planning. TR 98-11, Computer Science Dept., Iowa State University.
- Karaman, S. & Frazzoli, E. (2011). Sampling-based Algorithms for Optimal Motion Planning. International Journal of Robotics Research, 30(7), 846-894.

## Algorithm Overview

RRT* extends the basic RRT algorithm with two key improvements:

1. **Near neighbor rewiring**: When adding a new node, the algorithm searches for nearby nodes that could be reached more efficiently through the new node, and rewires their connections to reduce path cost.

2. **Asymptotic optimality**: As the number of samples approaches infinity, the cost of the solution converges to the optimal cost. The rewiring radius is adaptively computed as `gamma * (log(n)/n)^(1/d)` where *n* is the tree size and *d* is the dimensionality.

### Key Features
- **Probabilistically complete**: Given enough samples, will find a path if one exists
- **Asymptotically optimal**: Solution quality improves with more samples
- **Handles high-dimensional spaces**: No discretization required
- **Anytime algorithm**: Returns improving solutions over time

## Implementation Status

✅ **Fully Implemented** in `src/arco/planning/continuous/rrt.py`

### What's Implemented
- [x] Core RRT* algorithm with adaptive rewiring radius
- [x] Collision checking via `Occupancy` interface
- [x] Goal biasing for faster convergence
- [x] Early stopping option for first-solution mode
- [x] Configurable step size and sample limits
- [x] Tree export for visualization (`nodes_`, `edges_` attributes)

### Configuration Parameters

The `RRTPlanner` class accepts:

- `occupancy`: Occupancy map for collision checking
- `bounds`: Axis-aligned bounding box as `[(min, max), ...]` per dimension
- `max_sample_count`: Maximum iterations (default: 5000)
- `step_size`: Maximum extension step in world units (default: 1.0)
- `goal_tolerance`: Distance threshold for goal region (default: 0.5)
- `rewire_radius`: Search radius for rewiring (default: auto-computed)
- `collision_check_count`: Segment discretization for collision checks (default: 10)
- `goal_bias`: Probability of sampling goal directly (default: 0.05)
- `early_stop`: Stop at first solution vs. optimize further (default: False)

### Usage Example

```python
from arco.mapping import KDTreeOccupancy
from arco.planning.continuous import RRTPlanner
import numpy as np

# Create occupancy map from obstacle points
obstacles = np.array([[5, 5], [5, 6], [6, 5], [6, 6]])
occupancy = KDTreeOccupancy(obstacles, clearance=0.5)

# Configure planner
planner = RRTPlanner(
    occupancy=occupancy,
    bounds=[(0, 10), (0, 10)],
    max_sample_count=1000,
    step_size=0.5,
    goal_tolerance=0.3,
    goal_bias=0.05
)

# Plan path
start = np.array([1.0, 1.0])
goal = np.array([9.0, 9.0])
path = planner.plan(start, goal)

if path is not None:
    print(f"Found path with {len(path)} waypoints")
    # Access tree structure for visualization
    print(f"Tree has {len(planner.nodes_)} nodes")
```

## Comparison with SST

RRT* is designed for **geometric planning** (no dynamics), while SST handles **kinodynamic planning** (with dynamics constraints):

- **RRT\*** is simpler and faster for pure geometric problems
- **RRT\*** guarantees asymptotic optimality
- **SST** maintains a sparser tree and handles dynamics better
- **SST** is asymptotically near-optimal (not strictly optimal)

For purely geometric obstacle avoidance in ARCO, RRT* is the recommended choice.

## Visualization

The planner exposes tree structure through attributes:
- `nodes_`: List of node positions as numpy arrays
- `edges_`: List of (parent_index, child_index) tuples

Example visualization scripts are available in `tools/examples/rrt_planning.py`.

## Resources
- [Planning Algorithms, Ch. 5.5-5.6](http://planning.cs.uiuc.edu/ch5.pdf)
- [Original RRT* Paper](https://arxiv.org/abs/1105.1186)
- [Wikipedia: Rapidly-exploring Random Tree](https://en.wikipedia.org/wiki/Rapidly-exploring_random_tree)

---

*This document reflects the current RRT* implementation in ARCO.*
