"""Shared vehicle tracking helpers for the ARCO simulator.

Provides a unified :class:`VehicleConfig` dataclass and factory helpers
to build a :class:`~arco.guidance.vehicle.DubinsVehicle` with a
:class:`~arco.control.tracking.TrackingLoop` from any ordered
list of (x, y) waypoints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from arco.mapping.occupancy import Occupancy

from arco.config import load_config
from arco.control.joint_tracker import JointSpaceTracker
from arco.control.mpc import (
    DubinsPathFollowingMPC,
    DubinsVehicleLimits,
    JointSpaceMPC,
    JointSpaceMPCConfig,
    MPCTrackingLoop,
    PathFollowingMPCConfig,
)
from arco.control.pure_pursuit import PurePursuitController
from arco.control.tracking import TrackingLoop
from arco.guidance.vehicle import DubinsVehicle

# PPP / RRP joint-space MPC defaults when scenario YAML omits overrides.
# These scenarios use a finer 0.05 s control period than the global
# simulator.yml timestep (0.1 s); horizon dt MUST match that period.
DEFAULT_JOINT_HORIZON_STEP_COUNT: int = 12
DEFAULT_JOINT_HORIZON_DT: float = 0.05


def resolve_sim_timestep(
    cfg: dict[str, Any],
    *,
    global_sim_cfg: dict[str, Any] | None = None,
) -> float:
    """Return the simulation/control period from scenario or global config.

    Reads ``simulator.timestep`` or legacy ``simulator.dt`` from *cfg*
    first, then falls back to ``timestep`` in the global ``simulator.yml``.

    Args:
        cfg: Full scenario configuration dict (e.g. loaded map YAML).
        global_sim_cfg: Optional pre-loaded global simulator config.

    Returns:
        Control period in seconds.
    """
    sim = cfg.get("simulator", {})
    if isinstance(sim, dict):
        if sim.get("timestep") is not None:
            return float(sim["timestep"])
        if sim.get("dt") is not None:
            return float(sim["dt"])
    global_sim = global_sim_cfg or load_config("simulator")
    return float(global_sim["timestep"])


def path_following_mpc_config_from_simulator(
    sim_cfg: dict[str, Any] | None = None,
    *,
    cruise_speed: float | None = None,
    default_horizon_step_count: int | None = None,
    default_horizon_dt: float | None = None,
) -> PathFollowingMPCConfig:
    """Build path-following MPC config from global defaults + scenario YAML.

    Reads optional ``simulator.mpc.horizon.{step_count,dt}`` and
    ``simulator.mpc.weights.*`` overrides from *sim_cfg*.  When horizon
    keys are absent, *default_horizon_** values (if provided) replace the
    global ``mpc.yml`` horizon — used by the city demo to keep a longer
    anticipation window without changing other scenarios.

    Args:
        sim_cfg: Scenario ``simulator`` dict (may be ``None`` / empty).
        cruise_speed: Optional cruise override forwarded to
            :meth:`PathFollowingMPCConfig.create_from_config`.
        default_horizon_step_count: Horizon steps used when YAML omits
            ``mpc.horizon.step_count``.
        default_horizon_dt: Horizon ``dt`` used when YAML omits
            ``mpc.horizon.dt``.

    Returns:
        Configured :class:`PathFollowingMPCConfig`.
    """
    cfg = PathFollowingMPCConfig.create_from_config(cruise_speed=cruise_speed)
    sim = sim_cfg if isinstance(sim_cfg, dict) else {}
    mpc = sim.get("mpc") if isinstance(sim.get("mpc"), dict) else {}
    horizon = (
        mpc.get("horizon") if isinstance(mpc.get("horizon"), dict) else {}
    )
    weights = (
        mpc.get("weights") if isinstance(mpc.get("weights"), dict) else {}
    )
    step_count = horizon.get("step_count", default_horizon_step_count)
    dt = horizon.get("dt", default_horizon_dt)
    if step_count is not None or dt is not None:
        cfg = cfg.with_horizon_overrides(
            step_count=None if step_count is None else int(step_count),
            dt=None if dt is None else float(dt),
        )
    if weights:
        cfg = cfg.with_weight_overrides(
            contour=(
                None
                if weights.get("contour") is None
                else float(weights["contour"])
            ),
            heading=(
                None
                if weights.get("heading") is None
                else float(weights["heading"])
            ),
            progress=(
                None
                if weights.get("progress") is None
                else float(weights["progress"])
            ),
            lag=(
                None if weights.get("lag") is None else float(weights["lag"])
            ),
            control=(
                None
                if weights.get("control") is None
                else float(weights["control"])
            ),
            obstacle=(
                None
                if weights.get("obstacle") is None
                else float(weights["obstacle"])
            ),
            terminal=(
                None
                if weights.get("terminal") is None
                else float(weights["terminal"])
            ),
            contour_deadzone=(
                None
                if weights.get("contour_deadzone") is None
                else float(weights["contour_deadzone"])
            ),
        )
    return cfg


def joint_space_mpc_config_from_simulator(
    sim_cfg: dict[str, Any] | None = None,
    *,
    default_horizon_step_count: int | None = None,
    default_horizon_dt: float | None = None,
) -> JointSpaceMPCConfig:
    """Build joint-space MPC config from global defaults + scenario YAML.

    Reads optional ``simulator.mpc.horizon.{step_count,dt}`` and
    ``simulator.mpc.weights.*`` overrides from *sim_cfg*.  When horizon
    keys are absent, *default_horizon_** values (if provided) replace the
    global ``mpc.yml`` joint-space horizon.

    Args:
        sim_cfg: Scenario ``simulator`` dict (may be ``None`` / empty).
        default_horizon_step_count: Horizon steps when YAML omits
            ``mpc.horizon.step_count``.
        default_horizon_dt: Horizon ``dt`` when YAML omits
            ``mpc.horizon.dt``.

    Returns:
        Configured :class:`JointSpaceMPCConfig`.
    """
    cfg = JointSpaceMPCConfig.create_from_config()
    sim = sim_cfg if isinstance(sim_cfg, dict) else {}
    mpc = sim.get("mpc") if isinstance(sim.get("mpc"), dict) else {}
    horizon = (
        mpc.get("horizon") if isinstance(mpc.get("horizon"), dict) else {}
    )
    weights = (
        mpc.get("weights") if isinstance(mpc.get("weights"), dict) else {}
    )
    step_count = horizon.get("step_count", default_horizon_step_count)
    dt = horizon.get("dt", default_horizon_dt)
    if step_count is not None or dt is not None:
        cfg = cfg.with_horizon_overrides(
            step_count=None if step_count is None else int(step_count),
            dt=None if dt is None else float(dt),
        )
    if weights:
        cfg = replace(
            cfg,
            weight_tracking=float(
                weights.get("tracking", cfg.weight_tracking)
            ),
            weight_velocity=float(
                weights.get("velocity", cfg.weight_velocity)
            ),
            weight_control=float(weights.get("control", cfg.weight_control)),
            weight_obstacle=float(
                weights.get("obstacle", cfg.weight_obstacle)
            ),
        )
    return cfg


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
    # all other MPC fields (including lag / contour_deadzone) come from
    # mpc_cfg.  A manual field copy previously dropped progress-first keys,
    # so city YAML deadzone/lag never reached the race NMPC.
    effective_mpc_cfg = replace(mpc_cfg, cruise_speed=cfg.cruise_speed)
    tracker = DubinsPathFollowingMPC(
        vehicle_limits=limits,
        config=effective_mpc_cfg,
        occupancy=occupancy,
    )
    tracker.set_reference(waypoints)
    loop = MPCTrackingLoop(vehicle, tracker, cruise_speed=cfg.cruise_speed)
    return vehicle, loop


def build_joint_tracker(
    *,
    max_vel: float | list[float] | tuple[float, ...],
    max_acc: float | list[float] | tuple[float, ...],
    proportional_gain: float = 2.0,
    occupancy: Optional["Occupancy"] = None,
    repulsion_gain: float = 0.0,
    tracker: str = "pure_pursuit",
    mpc_cfg: JointSpaceMPCConfig | None = None,
) -> JointSpaceTracker | JointSpaceMPC:
    """Build a C-space tracker: P+APF or joint-space MPC.

    Args:
        max_vel: Per-axis velocity limits.
        max_acc: Per-axis acceleration limits.
        proportional_gain: P-gain for the APF tracker (ignored by MPC).
        occupancy: Optional C-space occupancy map.
        repulsion_gain: APF gain for the legacy tracker (ignored by MPC).
        tracker: ``"mpc"`` selects :class:`JointSpaceMPC`; any other
            value selects :class:`JointSpaceTracker`.
        mpc_cfg: Optional joint-space MPC config.

    Returns:
        A tracker exposing ``reset(q0)`` and ``step(target_q, dt)``.

    Raises:
        ImportError: If ``tracker == "mpc"`` and CasADi is missing.
    """
    if tracker == "mpc":
        return JointSpaceMPC(
            max_vel=max_vel,
            max_acc=max_acc,
            proportional_gain=proportional_gain,
            occupancy=occupancy,
            repulsion_gain=repulsion_gain,
            config=mpc_cfg or JointSpaceMPCConfig.create_from_config(),
        )
    return JointSpaceTracker(
        max_vel=max_vel,
        max_acc=max_acc,
        proportional_gain=proportional_gain,
        occupancy=occupancy,
        repulsion_gain=repulsion_gain,
    )
