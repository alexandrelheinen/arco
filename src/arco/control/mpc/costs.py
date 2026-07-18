"""Private cost helpers for path-following MPC."""

from __future__ import annotations

from typing import Any


def obstacle_barrier(
    distance: Any,
    clearance: float,
    weight: float,
    power: float,
    cone_factor: Any,
) -> Any:
    """Soft clearance barrier with forward-cone weighting.

    Matches the :class:`~arco.planning.continuous.optimizer.TrajectoryOptimizer`
    philosophy: penalize penetration of the clearance margin with a power
    barrier, then weight by directional relevance along the velocity cone.

    Args:
        distance: Distance to the nearest obstacle (symbolic or float).
        clearance: Required clearance radius (m). Must be positive.
        weight: Barrier weight.
        power: Barrier exponent.
        cone_factor: Forward-cone factor in ``[0, 1]`` (1 ahead, 0 behind).

    Returns:
        Scalar barrier cost (same type as *distance* / *cone_factor*).
    """
    safe_clearance = max(float(clearance), 1e-6)
    # penetration ratio: max(0, -(d - clearance) / clearance)
    # = max(0, (clearance - d) / clearance)
    penetration = (safe_clearance - distance) / safe_clearance
    # Use fmax for CasADi compatibility; also works for floats.
    try:
        import casadi as ca

        if isinstance(distance, (ca.SX, ca.MX, ca.DM)):
            penetration = ca.fmax(0.0, penetration)
            directional = 0.2 + 0.8 * cone_factor
            return weight * (penetration**power) * directional
    except ImportError:
        pass

    penetration_f = max(0.0, float(penetration))
    directional_f = 0.2 + 0.8 * float(cone_factor)
    return float(weight) * (penetration_f**power) * directional_f


def forward_cone_factor(
    pose_x: Any,
    pose_y: Any,
    heading: Any,
    obstacle_x: Any,
    obstacle_y: Any,
) -> Any:
    """Return ``max(0, cos(heading − bearing_to_obstacle))``.

    Args:
        pose_x: Vehicle x position.
        pose_y: Vehicle y position.
        heading: Vehicle heading (rad).
        obstacle_x: Obstacle x position.
        obstacle_y: Obstacle y position.

    Returns:
        Cone factor in ``[0, 1]`` (symbolic or float).
    """
    try:
        import casadi as ca

        if isinstance(pose_x, (ca.SX, ca.MX, ca.DM)):
            bearing = ca.atan2(obstacle_y - pose_y, obstacle_x - pose_x)
            return ca.fmax(0.0, ca.cos(heading - bearing))
    except ImportError:
        pass

    import math

    bearing = math.atan2(
        float(obstacle_y) - float(pose_y),
        float(obstacle_x) - float(pose_x),
    )
    return max(0.0, math.cos(float(heading) - bearing))
