# Guidance Layer Overview

The guidance layer in ARCO provides components for trajectory shaping and feedback control. After a planner produces a discrete or sampled path, the guidance layer refines it into a smooth, executable trajectory and tracks it with appropriate control laws.

## Architecture

Guidance owns interpolation, primitives, and vehicle models. Feedback
controllers live in `arco.control` (re-exported from `arco.guidance` for
convenience).

```
src/arco/guidance/
├── interpolation/    ← Path smoothing
├── primitive/        ← Kinematic exploration primitives
└── vehicle.py        ← Vehicle kinematic models

src/arco/control/
├── pid.py / pure_pursuit.py / tracking.py
└── mpc/              ← Path-following and joint-space MPC
```

## Components

### Control (`arco.control`)

Feedback controllers that generate control inputs to track a reference trajectory.

#### Implemented Controllers

- **PIDController** (`arco.control.pid`): Proportional-Integral-Derivative controller
  - Classic feedback control for setpoint tracking
  - Configurable gains (Kp, Ki, Kd)
  - Derivative filtering and anti-windup

- **PurePursuitController** (`arco.control.pure_pursuit`): Geometric path tracking
  - Look-ahead distance determines aggressiveness
  - Works well for car-like vehicles

- **DubinsPathFollowingMPC** (`arco.control.mpc`): Receding-horizon
  **contouring MPCC**.  Full mathematical description:
  [control_mpcc.md](control_mpcc.md).  Tunable knobs after planning
  (horizon, weights, dynamics, κ shaping): [control_tracking_params.md](control_tracking_params.md).
  - Classical MPCC structure: virtual progress speed `v_s` as a decision
    variable (`ṡ = v_s ≥ 0`), contouring / lag error split at `p(s)`,
    **linear** progress reward, `v_s` hard-capped by the curve-limited
    cruise `min(v_cruise, ω_max/|κ(s)|)`
  - Lag coupling is structural (`weight_lag > 0` enforced): it glues the
    virtual progress to the vehicle, so no projection blending or
    progress-monotonicity constraints are needed inside the NLP
  - CasADi + IPOPT backend via optional extra: `pip install arco[mpc]`
  - Cubic B-spline reference interpolants and a straight runway extension
    past the goal keep IPOPT convergent at polyline kinks and at the end
    of the path
  - Anti-stall: a parked warm start is replaced by a reference-rollout
    initial guess so the solver can escape stopped equilibria at sharp
    kinks (turn in place, then drive)
  - Soft directional obstacle barriers (not hard road tubes)
  - Paired with `MPCTrackingLoop`
  - Enable in SE(2) races with `simulator.tracker: mpc`; the model dt
    must equal the control period (city: 50 × 0.1 s = 5.0 s preview)
  - Trajectory evolution: on straights the car holds cruise near the
    reference; before kinks the capped progress speed brakes it to a
    lane-feasible corner speed, and `s` keeps increasing — no
    snap-turning, orbiting, parked stalls, or wall cuts
- **JointSpaceMPC** (`arco.control.mpc.joint_space`): N-DOF carrot-tracking NMPC
  - Drop-in for `JointSpaceTracker` (`reset` / `step` API)
  - Used by PPP / RRP when `tracker: mpc`

- **MPCTrackingLoop** / **TrackingLoop**: integration loops for MPC and Pure Pursuit
  - `TrackingLoop` accepts `VehicleModel` / `PathTracker` duck types and an
    optional `avoidance=` strategy (`AvoidanceStrategy`).  When omitted,
    `occupancy` + `repulsion_gain` build the default
    `ArtificialPotentialField` (same APF behaviour as before).
- **ArtificialPotentialField** (`arco.control.avoidance`): default APF
  turn-rate bias; construct with `occupancy=None` / `repulsion_gain<=0`
  for a no-op disable path.

- **MPCController**: Deprecated scalar stub — use `DubinsPathFollowingMPC`

PID / Pure Pursuit inherit from `Controller`. Path-following MPC uses `MPCTracker`.

### Interpolation (`arco.guidance.interpolation`)

Converts discrete waypoint paths into smooth, continuous trajectories.

#### Implemented Interpolators

- **BSplineInterpolator** (`interpolation/bspline.py`): currently a pass-through
  stub (`interpolate(path)` returns `path` unchanged). Intended for C² B-spline
  smoothing via scipy once implemented.

All interpolators inherit from `Interpolator` (`interpolation/base.py`).

### Primitives (`arco.guidance.primitive`)

Kinematic motion primitives for graph exploration and steering.

#### Implemented Primitives

- **DubinsPrimitive** (`primitive/dubins.py`): Dubins path steering
  - Shortest paths for car-like vehicles (forward-only, fixed turning radius)
  - Generates curved motion primitives for sampling-based planners
  - Based on Dubins (1957) optimal paths

All primitives inherit from the `ExplorationPrimitive` abstract base class (`primitive/base.py`).

**Wiring status:** RRT* / SST do **not** call `ExplorationPrimitive` by
default.  Default steering is geometric straight-line extension.  Pass a
primitive-backed callable via `RRTPlanner(..., steerer=...)` /
`SSTPlanner(..., steerer=...)` to enable kinodynamic expansion.  See
[PLANNING.md](PLANNING.md) § Extension Points.

### Vehicle Models (`arco.guidance.vehicle`)

Kinematic and dynamic models for different vehicle types.

#### Implemented Models

- **DubinsVehicle** (`vehicle.py`): Car-like kinematic model
  - Forward-only motion with minimum turning radius
  - Used with DubinsPrimitive and Pure Pursuit controller
  - Models unicycle-like kinematics

## Usage Patterns

### Path Smoothing Workflow

```python
from arco.guidance.interpolation import BSplineInterpolator
from arco.planning import AStar
import numpy as np

# 1. Plan discrete path (AStar wraps a numpy occupancy grid)
grid = np.zeros((20, 20), dtype=np.uint8)
planner = AStar(grid, grid_type="euclidean")
path = planner.search(start=(0, 0), goal=(19, 19))

# 2. Smooth with B-spline (currently a pass-through stub)
interpolator = BSplineInterpolator(degree=3)
smooth_path = interpolator.interpolate(path)
```

### Pure Pursuit Control Workflow

```python
from arco.control import PurePursuitController, TrackingLoop
from arco.guidance.vehicle import DubinsVehicle

vehicle = DubinsVehicle(x=0.0, y=0.0, heading=0.0, max_turn_rate=1.0)
controller = PurePursuitController(lookahead_distance=3.0)
loop = TrackingLoop(vehicle, controller, cruise_speed=0.5)
metrics = loop.run(trajectory_points, steps=100, dt=0.05)
```

### Path-following MPC Workflow

```python
from arco.control.mpc import (
    DubinsPathFollowingMPC,
    DubinsVehicleLimits,
    PathFollowingMPCConfig,
)
from arco.simulator.sim.tracking import VehicleConfig, build_vehicle_mpc_sim

cfg = VehicleConfig(
    max_speed=1.0,
    min_speed=0.05,
    cruise_speed=0.36,
    lookahead_distance=1.0,  # unused by MPC; kept for VehicleConfig parity
    goal_radius=0.2,
    max_turn_rate=1.2,
    max_acceleration=1.2,
    max_turn_rate_dot=2.0,
)
mpc_cfg = PathFollowingMPCConfig.create_from_config()
vehicle, loop = build_vehicle_mpc_sim(waypoints, cfg, mpc_cfg, occupancy=occ)
for _ in range(200):
    metrics = loop.step(waypoints, dt=0.05)
```

## Integration with Planning

The guidance layer operates downstream of the planning layer:

1. **Planning**: Produces feasible waypoint paths (discrete or sampled)
2. **Interpolation**: Smooths waypoints into continuous trajectories
3. **Control**: Tracks trajectories with feedback control

A typical workflow combines route planning (A*) → B-spline interpolation →
Pure Pursuit control (legacy alternative to MPC in SE(2) races)
(see [VISUALIZATION.md](VISUALIZATION.md)).

## References

- Snider, J. M. (2009). Automatic Steering Methods for Autonomous Automobile Path Tracking. Robotics Institute, CMU.
- Dubins, L. E. (1957). On Curves of Minimal Length with a Constraint on Average Curvature. American Journal of Mathematics.
- Rawlings, J. B., Mayne, D. Q., & Diehl, M. (2017). Model Predictive Control: Theory, Computation, and Design.

---

*This document reflects the current state of the guidance layer.*
