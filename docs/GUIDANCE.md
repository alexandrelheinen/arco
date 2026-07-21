# Guidance Layer Overview

The guidance layer in ARCO provides components for trajectory shaping and feedback control. After a planner produces a discrete or sampled path, the guidance layer refines it into a smooth, executable trajectory and tracks it with appropriate control laws.

## Architecture

The guidance layer is organized into three sub-packages:

```
src/arco/guidance/
├── __init__.py
├── control/          ← Feedback controllers and tracking loops
├── interpolation/    ← Path smoothing and trajectory generation
├── primitive/        ← Kinematic exploration primitives
└── vehicle.py        ← Vehicle kinematic models
```

## Components

### Control (`arco.guidance.control`)

Feedback controllers that generate control inputs to track a reference trajectory.

#### Implemented Controllers

- **PIDController** (`control/pid.py`): Proportional-Integral-Derivative controller
  - Classic feedback control for setpoint tracking
  - Configurable gains (Kp, Ki, Kd)
  - Derivative filtering and anti-windup

- **PurePursuitController** (`control/pure_pursuit.py`): Geometric path tracking
  - "Carrot-following" algorithm for smooth path tracking
  - Look-ahead distance determines aggressiveness
  - Works well for car-like vehicles

- **DubinsPathFollowingMPC** (`control/mpc/`): Receding-horizon contouring NMPC
  - Jointly optimizes lateral error, heading, speed, and obstacle clearance
  - CasADi + IPOPT backend via optional extra: `pip install arco[mpc]`
  - Soft directional obstacle barriers (same philosophy as `TrajectoryOptimizer`)
  - Paired with `MPCTrackingLoop` (no APF blend; avoidance is inside the NLP)
  - Enable in SE(2) races with `simulator.tracker: mpc`
    (`map/vehicle.yml`, `map/city.yml` — RRT* / SST / A* each keep their
    own planned waypoint reference; MPC does not replace the global planner)
  - City race may override `simulator.mpc.horizon` (default **3.0 s**) and
    draws the predicted XY polyline so anticipation is visible in videos

- **JointSpaceMPC** (`control/mpc/joint_space.py`): N-DOF carrot-tracking NMPC
  - Drop-in for `JointSpaceTracker` (`reset` / `step` API)
  - Soft C-space obstacle barriers; used by PPP / RRP when `tracker: mpc`
  - Factory: `build_joint_tracker(..., tracker="mpc")`

- **MPCTrackingLoop** (`control/mpc/tracking_loop.py`): Drop-in metrics parallel
  to `TrackingLoop` for the MPC tracker (`build_vehicle_mpc_sim` factory)

- **TrackingLoop** (`control/tracking.py`): Pure Pursuit integration framework
  - Optional APF repulsion for reactive avoidance (legacy / baseline)

- **MPCController** (`control/mpc/controller.py`): Deprecated scalar stub
  - Emits `DeprecationWarning`; use `DubinsPathFollowingMPC` instead

PID / Pure Pursuit inherit from the `Controller` abstract base class
(`control/base.py`). Path-following MPC uses the separate `MPCTracker` ABC.

### Interpolation (`arco.guidance.interpolation`)

Converts discrete waypoint paths into smooth, continuous trajectories.

#### Implemented Interpolators

- **BSplineInterpolator** (`interpolation/bspline.py`): B-spline curve fitting
  - Smooth C² continuous curves through waypoints
  - Configurable degree (cubic by default)
  - Parameterized by arc length for uniform speed profiles

All interpolators inherit from the `Interpolator` abstract base class (`interpolation/base.py`).

### Primitives (`arco.guidance.primitive`)

Kinematic motion primitives for graph exploration and steering.

#### Implemented Primitives

- **DubinsPrimitive** (`primitive/dubins.py`): Dubins path steering
  - Shortest paths for car-like vehicles (forward-only, fixed turning radius)
  - Generates curved motion primitives for sampling-based planners
  - Based on Dubins (1957) optimal paths

All primitives inherit from the `ExplorationPrimitive` abstract base class (`primitive/base.py`).

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

# 1. Plan discrete path
grid = ...  # Grid object
planner = AStar(grid)
path = planner.plan(start, goal)  # List of waypoints

# 2. Smooth with B-spline
interpolator = BSplineInterpolator(degree=3)
smooth_trajectory = interpolator.fit(np.array(path))

# 3. Sample at uniform intervals
t_samples = np.linspace(0, 1, 100)
trajectory_points = [interpolator.evaluate(t) for t in t_samples]
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
Pure Pursuit control, as demonstrated in the `city` and `vehicle` scenarios
(see [VISUALIZATION.md](VISUALIZATION.md)).

## References

- Snider, J. M. (2009). Automatic Steering Methods for Autonomous Automobile Path Tracking. Robotics Institute, CMU.
- Dubins, L. E. (1957). On Curves of Minimal Length with a Constraint on Average Curvature. American Journal of Mathematics.
- Rawlings, J. B., Mayne, D. Q., & Diehl, M. (2017). Model Predictive Control: Theory, Computation, and Design.

---

*This document reflects the current state of the guidance layer.*
