"""Shared vehicle tracking helpers for the ARCO simulator.

Provides a unified :class:`VehicleConfig` dataclass and factory helpers
to build a :class:`~arco.guidance.vehicle.DubinsVehicle` with a
:class:`~arco.control.tracking.TrackingLoop` from any ordered
list of (x, y) waypoints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from arco.mapping.occupancy import Occupancy

from arco.control.mpc import (
    DubinsPathFollowingMPC,
    DubinsVehicleLimits,
    MPCTrackingLoop,
    PathFollowingMPCConfig,
)
from arco.control.pure_pursuit import PurePursuitController
from arco.control.tracking import TrackingLoop
from arco.guidance.vehicle import DubinsVehicle


@dataclass
class VehicleConfig:
    """Parameters for the Dubins vehicle and pure-pursuit controller.

    Attributes:
        max_speed: Maximum vehicle speed in m/s.
        min_speed: Minimum vehicle speed in m/s.
        cruise_speed: Nominal tracking speed in m/s.
        lookahead_distance: Pure-pursuit lookahead distance in meters.
        goal_radius: Distance at which the goal is considered reached (m).
        max_turn_rate: Maximum turn rate in radians/s.
        max_acceleration: Maximum linear acceleration in m/s².
        max_turn_rate_dot: Maximum turn-rate derivative in rad/s².
        curvature_gain: Feed-forward curvature gain (0 = disabled).
        repulsion_gain: Obstacle-repulsion turn-rate gain for the tracking
            loop (rad/m).  ``0.0`` disables repulsion.
    """

    max_speed: float
    min_speed: float
    cruise_speed: float
    lookahead_distance: float
    goal_radius: float
    max_turn_rate: float
    max_acceleration: float
    max_turn_rate_dot: float
    curvature_gain: float = field(default=0.0)
    repulsion_gain: float = field(default=1.5)


def initial_heading(path: list[tuple[float, float]]) -> float:
    """Return heading in radians from path[0] toward path[1].

    Args:
        path: Ordered list of ``(x, y)`` waypoints.

    Returns:
        Heading angle in radians, or 0.0 if fewer than 2 points.
    """
    if len(path) < 2:
        return 0.0
    dx = path[1][0] - path[0][0]
    dy = path[1][1] - path[0][1]
    return math.atan2(dy, dx)


def find_lookahead(
    x: float,
    y: float,
    path: list[tuple[float, float]],
    distance: float,
) -> tuple[float, float]:
    """Return the lookahead point on *path* at least *distance* meters away.

    Args:
        x: Current x-position in world meters.
        y: Current y-position in world meters.
        path: Ordered list of ``(x, y)`` waypoints.
        distance: Minimum lookahead distance in meters.

    Returns:
        ``(x, y)`` of the lookahead target.
    """
    if not path:
        return (x, y)
    closest = min(
        range(len(path)),
        key=lambda i: math.hypot(path[i][0] - x, path[i][1] - y),
    )
    for pt in path[closest:]:
        if math.hypot(pt[0] - x, pt[1] - y) >= distance:
            return pt
    return path[-1]


def build_vehicle_sim(
    waypoints: list[tuple[float, float]],
    cfg: VehicleConfig,
    occupancy: Optional["Occupancy"] = None,
) -> tuple[DubinsVehicle, TrackingLoop]:
    """Create a Dubins vehicle and tracking loop initialized at waypoints[0].

    Args:
        waypoints: Ordered list of ``(x, y)`` path waypoints.
        cfg: Vehicle and controller configuration.
        occupancy: Optional occupancy map.  When provided and
            ``cfg.repulsion_gain > 0``, the tracking loop applies an
            APF obstacle-repulsion correction at each step, steering
            the vehicle away from nearby obstacles.

    Returns:
        Tuple of ``(vehicle, tracking_loop)``.
    """
    x0, y0 = waypoints[0]
    theta0 = initial_heading(waypoints)
    vehicle = DubinsVehicle(
        x=x0,
        y=y0,
        heading=theta0,
        max_speed=cfg.max_speed,
        min_speed=cfg.min_speed,
        max_turn_rate=cfg.max_turn_rate,
        max_acceleration=cfg.max_acceleration,
        max_turn_rate_dot=cfg.max_turn_rate_dot,
    )
    controller = PurePursuitController(
        lookahead_distance=cfg.lookahead_distance,
    )
    loop = TrackingLoop(
        vehicle,
        controller,
        cruise_speed=cfg.cruise_speed,
        curvature_gain=cfg.curvature_gain,
        occupancy=occupancy,
        repulsion_gain=cfg.repulsion_gain,
    )
    return vehicle, loop


def build_vehicle_mpc_sim(
    waypoints: list[tuple[float, float]],
    cfg: VehicleConfig,
    mpc_cfg: PathFollowingMPCConfig,
    occupancy: Optional["Occupancy"] = None,
) -> tuple[DubinsVehicle, MPCTrackingLoop]:
    """Create a Dubins vehicle and MPC tracking loop at waypoints[0].

    Parallel factory to :func:`build_vehicle_sim` that uses
    :class:`~arco.control.mpc.DubinsPathFollowingMPC` instead of
    Pure Pursuit + APF.

    Args:
        waypoints: Ordered list of ``(x, y)`` path waypoints.
        cfg: Vehicle dynamic limits and cruise speed.
        mpc_cfg: Path-following MPC horizon and weights.
        occupancy: Optional occupancy map used inside the optimizer
            for directional obstacle barriers.

    Returns:
        Tuple of ``(vehicle, mpc_tracking_loop)``.

    Raises:
        ImportError: If the optional CasADi dependency is missing
            (``pip install arco[mpc]``).
    """
    x0, y0 = waypoints[0]
    theta0 = initial_heading(waypoints)
    vehicle = DubinsVehicle(
        x=x0,
        y=y0,
        heading=theta0,
        max_speed=cfg.max_speed,
        min_speed=cfg.min_speed,
        max_turn_rate=cfg.max_turn_rate,
        max_acceleration=cfg.max_acceleration,
        max_turn_rate_dot=cfg.max_turn_rate_dot,
    )
    limits = DubinsVehicleLimits(
        max_speed=cfg.max_speed,
        min_speed=cfg.min_speed,
        max_turn_rate=cfg.max_turn_rate,
        max_acceleration=cfg.max_acceleration,
        max_turn_rate_dot=cfg.max_turn_rate_dot,
    )
    # VehicleConfig.cruise_speed is the authoritative cruise for the sim;
    # MPC weights / horizon come from mpc_cfg.
    effective_mpc_cfg = PathFollowingMPCConfig(
        horizon_step_count=mpc_cfg.horizon_step_count,
        dt=mpc_cfg.dt,
        cruise_speed=cfg.cruise_speed,
        weight_contour=mpc_cfg.weight_contour,
        weight_heading=mpc_cfg.weight_heading,
        weight_progress=mpc_cfg.weight_progress,
        weight_control=mpc_cfg.weight_control,
        weight_obstacle=mpc_cfg.weight_obstacle,
        obstacle_barrier_power=mpc_cfg.obstacle_barrier_power,
        weight_terminal=mpc_cfg.weight_terminal,
        weight_slack=mpc_cfg.weight_slack,
        max_solver_iter_count=mpc_cfg.max_solver_iter_count,
    )
    tracker = DubinsPathFollowingMPC(
        vehicle_limits=limits,
        config=effective_mpc_cfg,
        occupancy=occupancy,
    )
    tracker.set_reference(waypoints)
    loop = MPCTrackingLoop(vehicle, tracker, cruise_speed=cfg.cruise_speed)
    return vehicle, loop
