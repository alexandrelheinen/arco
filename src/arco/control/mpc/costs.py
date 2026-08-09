"""Shared soft-barrier cost helpers for path-following and joint-space MPC.

Used by :mod:`arco.control.mpc.path_following` and
:mod:`arco.control.mpc.joint_space` for clearance penetration penalties
(and forward-cone weighting on the SE(2) path-following controller).
"""

from __future__ import annotations

from typing import Any


def _is_casadi(value: Any) -> bool:
    """Return True if *value* is a CasADi SX, MX, or DM."""
    try:
        import casadi as ca

        return isinstance(value, (ca.SX, ca.MX, ca.DM))
    except ImportError:
        return False


def obstacle_barrier(
    distance: Any,
    clearance: Any,
    weight: float,
    power: float,
    cone_factor: Any,
) -> Any:
    """Soft clearance barrier with forward-cone weighting.

    Matches the inlined barrier used by path-following / joint-space MPC
    and the :class:`~arco.planning.continuous.optimizer.TrajectoryOptimizer`
    philosophy: penalize penetration of the clearance margin with a power
    barrier, then weight by directional relevance along the velocity cone.

    Joint-space MPC (no heading) passes ``cone_factor=1.0`` so the
    directional term is exactly ``1``.

    Args:
        distance: Distance to the nearest obstacle (symbolic or float).
        clearance: Required clearance radius (m). May be a CasADi Opti
            parameter. Must be positive at solve time.
        weight: Barrier weight.
        power: Barrier exponent.
        cone_factor: Forward-cone factor in ``[0, 1]`` (1 ahead, 0 behind).
            Use ``1.0`` when no cone weighting applies.

    Returns:
        Scalar barrier cost (same type family as *distance* /
        *cone_factor*).
    """
    if _is_casadi(distance) or _is_casadi(clearance):
        import casadi as ca

        # Match NLP inlining: (c - d) / max(c, eps), then clamp.
        penetration = (clearance - distance) / ca.fmax(clearance, 1e-6)
        penetration = ca.fmax(0.0, penetration)
        directional = 0.2 + 0.8 * cone_factor
        return weight * (penetration**power) * directional

    denom = max(float(clearance), 1e-6)
    penetration_f = max(0.0, (float(clearance) - float(distance)) / denom)
    directional_f = 0.2 + 0.8 * float(cone_factor)
    return float(weight) * (penetration_f**power) * directional_f


def forward_cone_factor(
    pose_x: Any,
    pose_y: Any,
    heading: Any,
    obstacle_x: Any,
    obstacle_y: Any,
) -> Any:
    """Smooth forward-cone factor via heading–bearing projection.

    Uses the unit obstacle offset projected onto the vehicle heading,
    clamped to ``[0, 1]``.  This matches the path-following MPC
    formulation (no ``atan2`` kinks in the NLP graph).

    Numerically equal to ``max(0, cos(heading − bearing))`` for nonzero
    separation.

    Args:
        pose_x: Vehicle x position.
        pose_y: Vehicle y position.
        heading: Vehicle heading (rad).
        obstacle_x: Obstacle x position.
        obstacle_y: Obstacle y position.

    Returns:
        Cone factor in ``[0, 1]`` (symbolic or float).
    """
    if _is_casadi(pose_x) or _is_casadi(obstacle_x):
        import casadi as ca

        dx = obstacle_x - pose_x
        dy = obstacle_y - pose_y
        dist = ca.sqrt(dx**2 + dy**2 + 1e-9)
        fwd = (ca.cos(heading) * dx + ca.sin(heading) * dy) / dist
        return ca.fmax(0.0, fwd)

    import math

    dx = float(obstacle_x) - float(pose_x)
    dy = float(obstacle_y) - float(pose_y)
    dist = math.sqrt(dx * dx + dy * dy + 1e-9)
    fwd = (
        math.cos(float(heading)) * dx + math.sin(float(heading)) * dy
    ) / dist
    return max(0.0, fwd)
