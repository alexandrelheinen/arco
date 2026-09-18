# ARCO Algorithm Reference

Core computational blocks in the ARCO library: calculation loops that are
algorithmically non-trivial or computationally expensive. This document
excludes data validation, I/O, formatting, and presentation code.

A class may contain more than one core block.

---

## 1. A\* Search (`AStarPlanner.plan_with_diagnostics`)

**File**: `crates/arco-planning/src/discrete/astar.rs`

Priority-queue best-first graph search. Frontier entries are
`(f, direction_penalty, h, insertion_counter, node)`:

| Key | Role |
|-----|------|
| `f = g + h` | Primary sort key |
| `direction_penalty` | Prefer straight continuation over turns |
| `h` | Secondary tie-breaker toward the goal |
| `insertion_counter` | FIFO among remaining ties |

Default heuristic: `graph.heuristic` when present (Euclidean on grids), else
`graph.distance`. After a path is found, `_simplify_path` collapses
colinear steps.

| Operation | Cost |
|-----------|------|
| Expand node | O(b) branch factor |
| Heap push/pop | O(log N) |
| Worst-case search | O(N · b · log N) |
| Path simplification | O(n) path length |

---

## 2. RRT\* Expansion (`RRTPlanner.plan` / `get_tree`)

**File**: `crates/arco-planning/src/continuous/rrt.rs`

```
for iteration in range(max_sample_count):
    x_rand  = goal (prob goal_bias) else uniform sample
    x_near  = nearest tree node to x_rand          # linear scan
    x_new   = steer(x_near, x_rand)
    if collision_free(x_near, x_new):
        near_set = nodes within rewire radius      # linear scan
        parent = min-cost parent in near_set
        add(x_new, parent); rewire near_set
```

Rewire radius follows the Karaman & Frazzoli schedule when not fixed.
`plan` and `get_tree` both contain the expansion loop.

| Per iteration | Cost |
|---------------|------|
| `_nearest` / `_near` | O(n) |
| Segment collision check | O(k) samples |
| Rewire | O(\|near\| · k) |
| Worst-case total | O(N² · k) |

---

## 3. SST Expansion (`SSTPlanner._run`)

**File**: `crates/arco-planning/src/continuous/sst.rs`

```
for iteration in range(max_sample_count):
    x_rand = goal (prob goal_bias) else uniform sample
    x_sel  = nearest *active* node                 # linear over active
    x_new  = steer(x_sel, x_rand)
    if collision_free(x_sel, x_new):
        w = nearest witness within witness_radius  # linear over witnesses
        if dominated by current representative: continue
        deactivate old rep; activate x_new
```

Witness cells keep only the cheapest active representative (sparsification).

| Per iteration | Cost |
|---------------|------|
| `_select_active` | O(\|active\|) |
| `_nearest_witness` | O(\|witnesses\|) |
| Collision check | O(k) |

---

## 4. Trajectory Optimizer (`TrajectoryOptimizer.optimize`)

**File**: `crates/arco-planning/src/continuous/optimizer.rs`

Two-stage numerical refinement of a reference path:

1. **Stage 1** — place interior waypoints on the reference, initialize
   segment durations `tᵢ ∝ Lᵢ / v_cruise`, optional IK commands.
2. **Stage 2** — an `argmin` quasi-Newton solve jointly adjusts durations
   and interior positions. Deviation A-08 records that this reaches a
   different local minimum than the `scipy` solve it replaced, so the cost
   achieved is comparable and the solution vector is not. The decision log
   also explains why the decision variable is the logarithm of a duration
   rather than the duration.

Composite cost (weights from `arco/config/optimizer.yml`):

```
J = w_time · T²
  + w_deviation · Σ |pᵢ − refᵢ|²
  + w_velocity  · Σ (speed_i − v_cruise)²
  + w_collision · Σ max(0, clearance − dist)²
```

Collision distances use `KDTreeOccupancy.query_distances`.

---

## 5. Trajectory Pruner (`TrajectoryPruner.prune`)

**File**: `crates/arco-planning/src/continuous/pruner.rs`

BFS over a shortcut graph: edge `(i, j)` exists iff `path[i]→path[j]` is
collision-free. Returns a minimum-hop subsequence. Consecutive planner
waypoints remain connectable, so a path always exists when the input was
feasible.

| Operation | Cost |
|-----------|------|
| Candidate edges | O(n²) |
| Each collision check | O(k) |
| Total | O(n² · k) |

---

## 6. Dubins Steering (`DubinsPrimitive`)

**File**: `crates/arco-guidance/src/primitive/dubins.rs`

Computes shortest CSC/CCC paths for a forward-only car with fixed turning
radius. Used as an exploration / steering primitive for kinodynamic-style
growth and for executable vehicle paths.

Complexity: O(1) candidate paths evaluated per steer query (constant number
of Dubins word types).

---

## 7. Control Loops

### PID (`PIDController`) — `crates/arco-control/src/pid.rs`

```
e = reference - state
integral += e · dt   (anti-windup)
derivative = filtered (e - e_prev) / dt
u = Kp·e + Ki·integral + Kd·derivative
```

O(1) per `control()` call.

### Pure Pursuit (`PurePursuitController`) — `crates/arco-control/src/pursuit.rs`

1. Find look-ahead point at distance `L_d` on the path.
2. Curvature `κ = 2 · sin(α) / L_d`.
3. Steering `δ = arctan(κ · wheelbase)`.

Look-ahead search is O(n_path) per call.

### Path-following / joint-space MPC

**Files**: `crates/arco-control/src/mpc/path_following.rs`,
`crates/arco-control/src/mpc/joint_space.rs`

Each control step solves a short sequence of convex quadratic programs,
linearized about the previous solution and solved by Clarabel; no optional
dependency is required.
Path-following MPC is a classical **MPCC contouring controller**: virtual
progress speed \(v_s\) with a curve-limited cap, contouring / lag error
split at \(p(s)\), linear progress reward, Dubins dynamics, soft obstacle
barriers.  Full equations and naming vs classical MPCC:
[control_mpcc.md](control_mpcc.md).  Post-plan tunable knobs (YAML weights,
horizon, vehicle limits, κ shaping):
[control_tracking_params.md](control_tracking_params.md).  Joint-space MPC
tracks a carrot in configuration space with soft obstacle barriers.

Dominant cost: building and solving the convex program per tick, up to a
few sequential iterations.

---

## 8. Occupancy Queries (`KDTreeOccupancy`)

**File**: `crates/arco-mapping/src/occupancy.rs`

A native KD-tree answers `is_occupied` and `query_distances`. Build is
O(n_obs · log n_obs); each query is O(log n_obs) average.

The tree replaced `scipy.spatial.cKDTree`, and the trade is measured: a
single query is about 25 times faster and a build is slower, reaching 3.5
times slower at fifty thousand obstacles. A planner builds once and queries
thousands of times, so a whole solve comes out far ahead; a caller that
rebuilds the map as the world changes is the workload where that accounting
would need checking.

---

## 9. Sub-optimized implementations and proposals

| Location | Current | Complexity | Proposed fix | Known how? |
|----------|---------|------------|--------------|------------|
| `RrtPlanner::nearest_index` | Linear scan | O(n) | Periodic rebuild of the crate's own KD-tree | Yes |
| `RrtPlanner::collect_near` | Linear scan | O(n) | KD-tree range query (same index) | Yes |
| `RRTPlanner.plan` vs `get_tree` | Duplicated expansion loop | n/a | Have `plan` delegate to `get_tree` | Yes (easy) |
| `SSTPlanner._select_active` | Linear over active set | O(\|active\|) | KD-tree over active positions | Yes |
| `SSTPlanner._nearest_witness` | Linear over witnesses | O(\|witnesses\|) | KD-tree over witness positions | Yes |
| `PurePursuitController.control` | Full path scan each tick | O(n_path) | Monotonic segment index | Yes (easy) |
| `CartesianGraph.find_nearest_node` | Linear over nodes | O(n) | Static KD-tree for road graphs | Yes (easy) |
| Segment collision sampling | Fixed `collision_check_count` | O(k) | Adaptive / AABB prefilter | Partially |
| `TrajectoryOptimizer::optimize` | Dense quasi-Newton with a differenced gradient | many evals | Warm-start; analytic Jacobian | Partially |
| Path-following / joint MPC | Rebuild + solve a QP sequence each step | QP solve/tick | Warm-start; longer control period | Partially |
| `BSplineInterpolator::interpolate` | Returns the path unchanged | n/a | Implement the smoothing | Yes (feature, not micro-opt) |

### Notes

- The crate's KD-tree is built once and is immutable. For growing RRT*/SST
  trees, rebuild every `M ≈ sqrt(N_max)` inserts to keep the amortized insert
  cost reasonable. The measured build cost above is what that rebuild
  interval has to pay for.
- SST witnesses only grow, so witness KD-trees only need rebuild-on-growth.
- Pure Pursuit and `plan`/`get_tree` dedup are the cheapest wins.
- MPC / optimizer improvements depend on solver backend choices more than on
  micro-optimizing Python loops; further backend changes are a follow-up on
  the [roadmap](ROADMAP.md), not a drop-in rewrite of the current
  Clarabel-based formulation.
- Unknown / out of scope here: GPU collision checking, exact continuous
  collision detection for curved Dubins segments, and distributed planning.

## Related docs

- [PLANNING.md](PLANNING.md) and per-algorithm notes under `docs/planning_*.md`
- [API.md](API.md) for the public symbols that wrap these blocks
- [ROADMAP.md](ROADMAP.md) for planned MPC backend follow-ups
