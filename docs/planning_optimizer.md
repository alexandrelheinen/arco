# Trajectory Optimisation

The `TrajectoryOptimizer` refines a reference path produced by a global
planner (RRT\*, SST) into a time-optimal trajectory.  It runs in two
stages and is model-agnostic: any robot model that provides an inverse
kinematics callable and a feasibility predicate can be plugged in.

---

## Motivation

Global sampling-based planners (RRT\*, SST) produce collision-free paths
but do not reason about time, speed profiles, or kinematic feasibility.
The `TrajectoryOptimizer` post-processes these paths to:

- Minimize total traversal time.
- Keep the trajectory close to the reference path.
- Penalise speeds that deviate significantly from the cruise speed.
- Penalise obstacle proximity using the KD-tree occupancy structure.

---

## Discretization

The trajectory is divided into **N segments**, where *N = len(reference_path) − 1*.
The independent variable is a dimensionless **progress** *s* that advances
from 0 to *N* (one unit per segment).  Within segment *i*, progress goes
from *i* to *i + 1*.

The bijection between progress and physical time is:

```
t(s) = Σ_{j<⌊s⌋} tⱼ + frac(s) · t_{⌊s⌋}
```

where *tᵢ* is the traversal time of segment *i*.  This formulation keeps
the parametrisation decoupled from time, allowing the optimizer to freely
adjust the speed profile.

---

## Cost Function

The optimizer minimises the composite cost:

```
J = w_time · T²
  + w_deviation · Σᵢ |pᵢ − refᵢ|²
  + w_velocity  · Σᵢ (speed_i − v_cruise)²
  + w_collision · Σᵢ max(0, clearance − dist(pᵢ, obstacles))²
```

| Term | Symbol | Description |
|------|--------|-------------|
| Total time squared | `w_time · T²` | Drives the optimizer toward shorter trajectories. Dominant term by default. |
| Reference deviation | `w_deviation · Σ d²` | Keeps the optimized path close to the global planner output. |
| Velocity penalty | `w_velocity · Σ (v − v_cruise)²` | Prevents degenerate *T → 0* solutions; keeps speed near cruise. |
| Collision penalty | `w_collision · Σ c(p, obs)` | Penalises obstacle penetration using the KD-tree nearest-obstacle query. |

All four weights are configurable in `tools/config/optimizer.yml`.

---

## Two-Stage Solver

### Stage 1 — Initialisation (Inverse Kinematics)

An initial guess is built by:

- Placing each interior waypoint directly on the reference path (zero
  deviation).
- Setting segment durations: `tᵢ⁰ = α · Lᵢ / v_cruise`, where *Lᵢ* is
  the straight-line distance between consecutive waypoints and *α* is the
  `time_relaxation` factor (default 1.5 → 50 % slack).
- If an `inverse_kinematics` callable is provided, it is called as
  `ik(start, goal, speed, duration) → commands` to produce initial
  control commands.

This places the initial candidate in the correct topological region of
attraction and close to the local minimum, making Stage 2 fast and
reliable.

### Stage 2 — Local Refinement (`scipy.optimize.minimize`)

The Stage-1 candidate is passed to `scipy.optimize.minimize` using
`L-BFGS-B` (default) or `SLSQP`.  The optimizer jointly adjusts:

- The *N* segment durations `tᵢ` (bounded below by a small positive
  constant).
- The *N − 1* interior waypoint positions (unconstrained; the deviation
  cost acts as a soft bound).

---

## Feasibility Interface

Each robot model implements:

```python
def is_feasible(self, state: np.ndarray) -> bool:
    """Return True if state satisfies all dynamic constraints."""
```

For `DubinsVehicle`, `state` can be:
- 3 elements `(x, y, θ)` — always feasible (kinematic state only).
- 5 elements `(x, y, θ, speed, turn_rate)` — checked against
  `max_speed`, `min_speed`, and `max_turn_rate`.

The optimizer calls `feasibility(state)` for each optimized waypoint and
issues a `RuntimeWarning` for any infeasible state.  The result is still
returned; feasibility is a diagnostic, not a hard constraint at this stage.

---

## Robot Model Interface

The optimizer is model-agnostic and communicates with the robot model
through two optional callables:

| Callable | Signature | Purpose |
|----------|-----------|---------|
| `inverse_kinematics` | `(start, goal, speed, duration) → np.ndarray` | Returns control commands for Stage-1 initialization. |
| `feasibility` | `(state) → bool` | Post-optimization feasibility check. |

`DubinsVehicle` already implements both:

```python
vehicle = DubinsVehicle(max_speed=5.0)
result = optimizer.optimize(
    reference_path,
    inverse_kinematics=vehicle.inverse_kinematics,
    feasibility=vehicle.is_feasible,
)
```

---

## Return Value

`TrajectoryOptimizer.optimize()` returns a `TrajectoryResult` dataclass:

```python
@dataclass
class TrajectoryResult:
    states:    List[np.ndarray]  # N+1 positions (start + N waypoints)
    commands:  List[np.ndarray]  # N control-command vectors
    durations: List[float]       # N segment traversal times (s)
    cost:      float             # Final composite cost
```

---

## Configuration

All tuning parameters live in `tools/config/optimizer.yml`:

```yaml
cruise_speed: 1.0       # Target speed (world units / s)
weight_time: 10.0       # Time cost weight (dominant)
weight_deviation: 1.0   # Deviation cost weight
weight_velocity: 1.0    # Velocity cost weight
weight_collision: 5.0   # Collision cost weight
time_relaxation: 1.5    # α — 50 % slack on initial times
method: L-BFGS-B        # scipy optimizer method
sample_count: 3         # Intermediate collision samples per segment
```

---

## Usage Example

```python
from arco.mapping import KDTreeOccupancy
from arco.planning.continuous import RRTPlanner, TrajectoryOptimizer
from arco.guidance.vehicle import DubinsVehicle

# Build occupancy and plan a reference path
occ = KDTreeOccupancy(obstacle_points, clearance=0.5)
rrt = RRTPlanner(occ, bounds=[(0, 50), (0, 50)])
reference_path = rrt.plan(start, goal)

# Optimize the trajectory
vehicle = DubinsVehicle(max_speed=5.0)
optimizer = TrajectoryOptimizer(occ, cruise_speed=2.0)
result = optimizer.optimize(
    reference_path,
    inverse_kinematics=vehicle.inverse_kinematics,
    feasibility=vehicle.is_feasible,
)

print(f"Total time: {sum(result.durations):.2f} s")
print(f"Final cost: {result.cost:.4f}")
```

A fully worked visualization example is in
`tools/examples/trajectory_optimization.py`.

---

## References

- Ratliff, N., Zucker, M., Bagnell, J. A. & Srinivasa, S. (2009).
  CHOMP: Gradient optimization techniques for efficient motion planning.
  *ICRA 2009*.
- `scipy.optimize.minimize` documentation.
- Project architecture: `docs/guidelines.md`.
