# ARCO Algorithm Reference

This document identifies and documents the **core computational blocks**
(algorithms) in the ARCO library, and analyses their current performance
characteristics and known optimization opportunities.

A *core block* is a calculation loop that is computationally expensive or
algorithmically non-trivial.  It excludes data validation, I/O, formatting,
and presentation code.

---

## 1. A\* Search (`AStarPlanner.plan_with_diagnostics`)

**File**: `src/arco/planning/discrete/astar.py`

### Algorithm

Priority-queue (min-heap) best-first graph search.  Each node in the frontier
is stored as a 5-tuple `(f, direction_penalty, h, insertion_counter, node)`:

| Key | Role |
|-----|------|
| `f = g + h` | Total estimated cost — primary sort key |
| `direction_penalty` | 0 = same direction as previous, 1 = turn — tie-breaker |
| `h` | Heuristic estimate to goal — secondary tie-breaker |
| `insertion_counter` | FIFO among ties — prevents node-comparison errors |

The heuristic is the Euclidean distance between cell positions (`grid.heuristic`),
which is admissible on both `ManhattanGrid` and `EuclideanGrid`.

After the path is found, `_simplify_path` collapses consecutive steps with the
same direction into a single segment (reduces waypoint count with no cost change).

### Complexity

| Operation | Cost |
|-----------|------|
| Node expansion | O(b) per node where b = branch factor |
| Priority-queue push/pop | O(log N) where N = frontier size |
| Total (worst case) | O(N · b · log N) |
| `_simplify_path` | O(n) where n = raw path length |

### Known Issues

None. A* is the correct algorithm for the discrete planning use case and the
implementation is complete.

---

## 2. RRT\* Expansion Loop (`RRTPlanner.plan` / `RRTPlanner.get_tree`)

**File**: `src/arco/planning/continuous/rrt.py`

### Algorithm

```
for iteration in range(max_sample_count):
    x_rand  = goal  (with prob goal_bias)  OR  uniform sample in bounds
    x_near  = nearest tree node to x_rand          # O(n) linear scan
    x_new   = steer(x_near, x_rand)                # clamp to step_size
    if collision_free(x_near, x_new):              # O(k) samples on segment
        radius = gamma * (log(n)/n)^(1/d)          # Karaman & Frazzoli formula
        near_set = nodes within radius of x_new    # O(n) linear scan
        parent   = min-cost parent in near_set
        add(x_new, parent)
        rewire: for each near node, lower cost through x_new if possible
        check if x_new is within goal_tolerance of goal
```

**Code duplication**: the expansion loop is written out twice — once in `plan`
and once in `get_tree`.  Both implementations are functionally identical.

### Complexity

| Operation | Cost per iteration |
|-----------|-------------------|
| Nearest-node query (`_nearest`) | O(n) |
| Near-node query (`_near`) | O(n) |
| Segment collision check (`_segment_free`) | O(k) where k = `collision_check_count` |
| Rewiring (near nodes) | O(\|near\| · k) ≤ O(n · k) |
| **Per iteration total** | **O(n · k)** |
| **Total (worst case)** | **O(N² · k)** where N = `max_sample_count` |

### ⚠ Sub-optimal implementations (see §7)

- `_nearest`: O(n) linear scan → should use a KD-tree for O(log n)
- `_near`: O(n) linear scan → should use a KD-tree range query for O(log n + \|near\|)
- Duplicate expansion code in `plan` and `get_tree`

---

## 3. SST Expansion Loop (`SSTPlanner._run`)

**File**: `src/arco/planning/continuous/sst.py`

### Algorithm

```
for iteration in range(max_sample_count):
    x_rand     = goal  (with prob goal_bias)  OR  uniform sample
    x_selected = nearest *active* node to x_rand       # O(|active|) linear scan
    x_new      = steer(x_selected, x_rand)
    if collision_free(x_selected, x_new):
        new_cost = cost[x_selected] + dist(x_selected, x_new)
        w = nearest witness within witness_radius of x_new  # O(|witnesses|) scan
        if w is None:
            create new witness at x_new
        elif new_cost >= cost[current_rep(w)]:
            continue                          # dominated by existing representative
        deactivate current_rep(w)             # sparsification
        add x_new as new active node and rep for w
        check if x_new reaches goal
```

The witness mechanism keeps the tree *sparse*: each witness cell retains only
the cheapest-to-reach active node, discarding dominated ones.

### Complexity

| Operation | Cost per iteration |
|-----------|-------------------|
| Active-node selection (`_select_active`) | O(\|active\|) |
| Witness lookup (`_nearest_witness`) | O(\|witnesses\|) |
| Segment collision check (`_segment_free`) | O(k) |
| **Per iteration total** | **O(\|active\| + \|witnesses\| + k)** |

In the worst case \|active\| ≤ n and \|witnesses\| ≤ n, so the total is
O(N · (n + k)) where n is the current tree size.

### ⚠ Sub-optimal implementations (see §7)

- `_select_active`: O(\|active\|) linear scan → should use a KD-tree over active node positions
- `_nearest_witness`: O(\|witnesses\|) linear scan → should use a KD-tree over witness positions

---

## 4. Trajectory Optimizer (`TrajectoryOptimizer.optimize`)

**File**: `src/arco/planning/continuous/optimizer.py`

### Algorithm

Two-stage numerical optimization:

**Stage 1 — Initialization**

Builds an initial guess by:

1. Placing each interior waypoint on the reference path (zero deviation).
2. Computing initial segment durations: `tᵢ = α · Lᵢ / v_cruise` (arc-length
   proportional with relaxation factor α, default 1.5).
3. Optionally calling `inverse_kinematics(start, goal, speed, duration)` to
   initialize control commands.

**Stage 2 — Local refinement**

Calls `scipy.optimize.minimize` with method `L-BFGS-B` (default) or `SLSQP`.
The optimizer jointly adjusts:

- N segment durations `tᵢ` (bounded below by a small positive constant).
- (N−1) interior waypoint positions.

**Cost function** evaluated per call:

```
J = w_time · T²
  + w_deviation · Σᵢ |pᵢ − refᵢ|²
  + w_velocity  · Σᵢ (speed_i − v_cruise)²
  + w_collision · Σᵢ max(0, clearance − dist(pᵢ, obstacles))²
```

The collision term calls `KDTreeOccupancy.query_distances` which is an O(n_pts · log(n_obs))
KD-tree query.

### Complexity

| Operation | Cost |
|-----------|------|
| Cost function evaluation | O(N · n_obs_samples · log(n_obstacles)) |
| `scipy.optimize.minimize` calls | depends on convergence; typically O(100) function evaluations |

### Known Issues

None for the optimization framework itself. The quality of the solution depends
heavily on the weight tuning in `tools/config/optimizer.yml`.

---

## 5. Trajectory Pruner (`TrajectoryPruner.prune`)

**File**: `src/arco/planning/continuous/pruner.py`

### Algorithm

BFS over a path-shortcut graph to find the minimum-hop subsequence:

```
graph vertices: path node indices {0, …, n−1}
graph edges:    (i, j) iff direct segment path[i]→path[j] is collision-free
BFS from 0 to n−1 → minimum-waypoint path
```

Because consecutive nodes in the original planner path are always
directly connectable (guaranteed by the planner), a path always exists.

The invariant is enforced by `_is_segment_feasible` using the same
`_segment_free` collision check as the planner.

### Complexity

| Operation | Cost |
|-----------|------|
| Candidate edge checks | O(n²) in the worst case |
| Each collision check | O(k) where k = `collision_check_count` |
| **Total** | **O(n² · k)** |

For typical paths produced by RRT* (n ≤ 50 nodes after step-size filtering),
this is approximately 50 × 50 × 10 = 25,000 collision queries, which is fast.
For very long paths (n > 500), this can become a bottleneck.

### Known Issues

The BFS is already optimal (minimum hop count).  The O(n²) edge enumeration
is theoretically unavoidable for an optimal pruner.  The constant factor (k)
can be reduced by lowering `collision_check_count` at the cost of safety.

---

## 6. B-Spline Interpolator (`BSplineInterpolator.interpolate`)

**File**: `src/arco/guidance/interpolation/bspline.py`

### Current Status: Stub

The current implementation is a **pass-through stub** — it returns the input
path unchanged:

```python
def interpolate(self, path):
    return path   # Placeholder; B-spline not yet implemented
```

No computation is performed.

### Intended Algorithm

The planned implementation would:
1. Parameterize the path by arc length.
2. Fit a B-spline of degree k through the waypoints via `scipy.interpolate.splprep`.
3. Evaluate the spline at uniform parameter values via `scipy.interpolate.splev`.

This would produce a C² continuous curve with controllable curvature, suitable
for non-holonomic vehicle trajectory tracking.

---

## 7. Control Loops

### PID Controller (`PIDController`)

**File**: `src/arco/control/pid.py`

Classic PID feedback control law:

```
error_i  = reference - state
integral += error_i * dt  (with anti-windup clamping)
derivative = (error_i - prev_error) / dt  (with low-pass filter)
output = Kp * error_i + Ki * integral + Kd * derivative
```

**Complexity**: O(1) per `control()` call.

### Pure Pursuit Controller (`PurePursuitController`)

**File**: `src/arco/control/pure_pursuit.py`

Geometric path-tracking algorithm:

```
1. Find the look-ahead point: nearest path point at distance L_d ahead
   (circle-segment intersection)
2. Compute the signed curvature: κ = 2 * sin(α) / L_d
   where α = angle from vehicle heading to the look-ahead point
3. Steering angle: δ = arctan(κ * wheelbase)
```

**Complexity**: O(n_path) per `control()` call (scan for look-ahead point).

### ⚠ Sub-optimal implementation

The look-ahead scan is a linear scan over all path segments: O(n_path).
For long paths, a segment index pointer that advances monotonically would
reduce this to O(1) amortized.

---

## 8. Summary of Optimization Opportunities

| Location | Current | Complexity | Proposed Fix | Difficulty |
|----------|---------|------------|--------------|------------|
| `RRTPlanner._nearest` | Linear scan | O(n) | scipy KD-tree (batch rebuild every M inserts) | Medium |
| `RRTPlanner._near` | Linear scan | O(n) | KD-tree range query | Medium (same KD-tree as `_nearest`) |
| `RRTPlanner.get_tree` | Duplicate of `plan` loop | n/a | Refactor `plan` to call `get_tree` and extract path | Easy |
| `SSTPlanner._select_active` | Linear scan over active set | O(\|active\|) | KD-tree over active positions, rebuilt on deactivations | Medium |
| `SSTPlanner._nearest_witness` | Linear scan over witnesses | O(\|witnesses\|) | KD-tree over witness positions | Medium (simpler than active, rarely deactivated) |
| `PurePursuitController.control` | Linear path scan | O(n_path) | Monotonically advancing index pointer | Easy |
| `BSplineInterpolator.interpolate` | Stub, no-op | n/a | Implement with `scipy.interpolate.splprep/splev` | Easy–Medium |
| `CartesianGraph.find_nearest_node` | Linear scan | O(n) | KD-tree (static for road networks) | Easy |

### Notes

- All the KD-tree proposals require `scipy.spatial.KDTree` (already in the
  dependency tree via scipy).  The main challenge is managing incremental
  inserts for the RRT*/SST trees: `scipy.spatial.KDTree` is immutable, so
  the tree must be rebuilt periodically (every M inserts).  An
  `M = sqrt(N_max)` rebuild schedule keeps amortized insert cost at
  O(sqrt(N_max) · log N_max).

- For `SSTPlanner` the witness list rarely shrinks (only new witnesses are
  added, never removed), so the KD-tree needs only periodic rebuilds on
  growth — a simpler operation.

- The `PurePursuitController` optimization is the easiest: a single integer
  index tracking the last-matched segment.

- The `BSplineInterpolator` stub is not sub-optimal — it does nothing.  Its
  priority should be based on whether smooth trajectory tracking is needed
  for any planned scenario.
