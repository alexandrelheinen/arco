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

- **MPCController** (`control/mpc.py`): Model Predictive Control
  - Optimization-based control with preview horizon
  - Handles constraints and multi-objective cost
  - More computationally expensive than PID/Pure Pursuit

- **TrackingLoop** (`control/tracking.py`): Integration framework
  - Wraps a controller with execution timing and state management
  - Provides consistent interface for all controller types

All controllers inherit from the `Controller` abstract base class (`control/base.py`).

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
from arco.guidance.control import PurePursuitController, TrackingLoop
from arco.guidance.vehicle import DubinsVehicle

# 1. Create vehicle model
vehicle = DubinsVehicle(turning_radius=2.0)

# 2. Configure controller
controller = PurePursuitController(
    lookahead_distance=3.0,
    vehicle=vehicle
)

# 3. Set reference path
controller.set_path(trajectory_points)

# 4. Execute tracking loop
loop = TrackingLoop(controller, dt=0.1)
for state in loop.run():
    # Apply control and update vehicle
    control_input = loop.compute_control(state)
    # ... apply to vehicle physics
```

## Integration with Planning

The guidance layer operates downstream of the planning layer:

1. **Planning**: Produces feasible waypoint paths (discrete or sampled)
2. **Interpolation**: Smooths waypoints into continuous trajectories
3. **Control**: Tracks trajectories with feedback control

For example, the **Horse Auto-Follow** system (see [horse_auto_follow.md](horse_auto_follow.md)) uses:
- Route planning (A*) → waypoint path on road graph
- B-spline interpolation → smooth trajectory
- Pure Pursuit control → steering commands

## References

- Snider, J. M. (2009). Automatic Steering Methods for Autonomous Automobile Path Tracking. Robotics Institute, CMU.
- Dubins, L. E. (1957). On Curves of Minimal Length with a Constraint on Average Curvature. American Journal of Mathematics.
- Rawlings, J. B., Mayne, D. Q., & Diehl, M. (2017). Model Predictive Control: Theory, Computation, and Design.

---

*This document reflects the current state of the guidance layer.*
