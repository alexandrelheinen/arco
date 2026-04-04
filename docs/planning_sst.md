# SST Path Planning

SST (Stable Sparse Trees) is an asymptotically near-optimal sampling-based motion planning algorithm designed for kinodynamic planning. It maintains a sparse representative subset of nodes to balance exploration quality with memory efficiency.

## Key Reference
- Li, Y., Littlefield, Z., & Bekris, K. E. (2016). Asymptotically Optimal Sampling-based Kinodynamic Planning. International Journal of Robotics Research, 35(5), 528-564.

## Algorithm Overview

SST addresses the challenge of kinodynamic planning (planning with dynamics constraints) by maintaining a **sparse witness tree**:

1. **Witness cells**: The state space is implicitly partitioned into witness cells with radius δ_s (witness_radius). Each cell keeps at most one "active" representative node.

2. **Sparsification**: When a new node is added, if it falls into an existing witness cell and has lower cost-to-come than the current representative, the old representative is deactivated and the new one becomes active.

3. **Best-first propagation**: Only active nodes can be extended, keeping the tree compact while maintaining coverage.

### Key Features
- **Asymptotically near-optimal**: Converges to a solution within a factor of the optimal cost
- **Memory efficient**: Sparse representation scales better than dense trees
- **Handles kinodynamic constraints**: Designed for systems with dynamics
- **Anytime algorithm**: Progressive refinement of solutions

## Implementation Status

✅ **Fully Implemented** in `src/arco/planning/continuous/sst.py`

### What's Implemented
- [x] Core SST algorithm with witness-based sparsification
- [x] Active/inactive node tracking
- [x] Collision checking via `Occupancy` interface
- [x] Goal biasing for faster convergence
- [x] Early stopping option
- [x] Configurable witness radius and step size
- [x] Tree export for visualization (`nodes_`, `edges_`, `active_` attributes)

### Configuration Parameters

The `SSTPlanner` class accepts:

- `occupancy`: Occupancy map for collision checking
- `bounds`: Axis-aligned bounding box as `[(min, max), ...]` per dimension
- `max_sample_count`: Maximum propagation attempts (default: 5000)
- `step_size`: Maximum extension step in world units (default: 1.0)
- `goal_tolerance`: Distance threshold for goal region (default: 0.5)
- `witness_radius`: Witness cell half-width δ_s (default: 0.8)
  - **Critical**: Must be less than `step_size` for tree growth
  - Smaller → denser tree → better optimality, higher memory
- `collision_check_count`: Segment discretization for collision checks (default: 10)
- `goal_bias`: Probability of sampling goal directly (default: 0.05)
- `early_stop`: Stop at first solution vs. optimize further (default: False)

### Usage Example

```python
from arco.mapping import KDTreeOccupancy
from arco.planning.continuous import SSTPlanner
import numpy as np

# Create occupancy map from obstacle points
obstacles = np.array([[5, 5], [5, 6], [6, 5], [6, 6]])
occupancy = KDTreeOccupancy(obstacles, clearance=0.5)

# Configure planner
# Note: witness_radius < step_size is required
planner = SSTPlanner(
    occupancy=occupancy,
    bounds=[(0, 10), (0, 10)],
    max_sample_count=1000,
    step_size=1.0,
    witness_radius=0.8,  # Must be < step_size
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
    active_count = sum(planner.active_)
    print(f"Tree has {len(planner.nodes_)} total nodes, {active_count} active")
```

## ARCO's Geometric SST Variant

The ARCO implementation of SST targets **purely geometric planning** (no dynamics), so the propagation step uses simple straight-line steering toward random samples, similar to RRT*. This differs from the original SST paper which focuses on kinodynamic systems.

### When to Use SST vs. RRT*

**Use RRT\* when:**
- Pure geometric obstacle avoidance is needed
- Asymptotic optimality guarantee is required
- Memory is not a constraint

**Use SST when:**
- Memory efficiency is important (sparse tree)
- You want to experiment with witness-based sparsification
- Future extension to kinodynamic planning is anticipated

For most geometric planning tasks in ARCO, **RRT\* is the recommended default**.

## Sparse Competition Feature

Recent commits added "sparse competition" between RRT* and SST, allowing comparative benchmarking of both algorithms on the same problem instances.

## Visualization

The planner exposes tree structure through attributes:
- `nodes_`: List of node positions as numpy arrays
- `edges_`: List of (parent_index, child_index) tuples
- `active_`: Boolean array indicating which nodes are active witnesses

Example visualization scripts are available in `tools/examples/sst_planning.py`.

## Resources
- [Original SST Paper](https://arxiv.org/abs/1407.2896)
- [Planning Algorithms, Ch. 14](http://planning.cs.uiuc.edu/ch14.pdf)
- [Comparison: RRT* vs. SST](https://robotics.stackexchange.com/questions/19592)

---

*This document reflects the current SST implementation in ARCO.*
