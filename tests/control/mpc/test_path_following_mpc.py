"""Functional tests for DubinsPathFollowingMPC."""

from __future__ import annotations

import math

import pytest

casadi = pytest.importorskip("casadi")

from arco.control.mpc.path_following import (  # noqa: E402
    DubinsPathFollowingMPC,
    DubinsVehicleLimits,
    PathFollowingMPCConfig,
)
from arco.guidance.vehicle import DubinsVehicle  # noqa: E402
from tests.control.mpc.conftest import RectOccupancy  # noqa: E402


def _limits(
    *,
    max_speed: float = 1.0,
    min_speed: float = 0.05,
    max_turn_rate: float = 1.2,
    max_acceleration: float = 1.5,
    max_turn_rate_dot: float = 2.0,
) -> DubinsVehicleLimits:
    return DubinsVehicleLimits(
        max_speed=max_speed,
        min_speed=min_speed,
        max_turn_rate=max_turn_rate,
        max_acceleration=max_acceleration,
        max_turn_rate_dot=max_turn_rate_dot,
    )


def _cfg(**overrides) -> PathFollowingMPCConfig:
    base = PathFollowingMPCConfig(
        horizon_step_count=20,
        dt=0.05,
        cruise_speed=0.6,
        weight_contour=12.0,
        weight_heading=6.0,
        weight_progress=2.0,
        weight_control=0.05,
        weight_obstacle=80.0,
        obstacle_barrier_power=4.0,
        weight_terminal=25.0,
        max_solver_iter_count=80,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_mpc_tracks_straight_path_no_obstacles(straight_path) -> None:
    cfg = _cfg(cruise_speed=0.5)
    mpc = DubinsPathFollowingMPC(vehicle_limits=_limits(), config=cfg)
    mpc.set_reference(straight_path)

    vehicle = DubinsVehicle(
        x=0.0,
        y=0.05,
        heading=0.0,
        max_speed=cfg.cruise_speed + 0.2,
        min_speed=0.05,
        max_turn_rate=1.2,
        max_acceleration=1.5,
        max_turn_rate_dot=2.0,
    )
    # Seed near cruise so the horizon is useful immediately.
    vehicle._speed = cfg.cruise_speed

    dt = cfg.dt
    steps = int(5.0 / dt)
    success_count = 0
    for _ in range(steps):
        result = mpc.step(
            vehicle.pose,
            speed=vehicle.speed,
            turn_rate=vehicle.turn_rate,
            dt=dt,
        )
        if result.solver_success:
            success_count += 1
        vehicle.step(result.speed_cmd, result.turn_rate_cmd, dt)

    assert abs(result.cross_track_error) < 0.05
    assert success_count / steps > 0.95
    assert len(result.predicted_xy) == cfg.horizon_step_count + 1
    assert all(len(pt) == 2 for pt in result.predicted_xy)


def test_mpc_slows_before_box_obstacle(straight_path) -> None:
    # Box straddles the path ahead; MPC must decelerate before impact.
    occ = RectOccupancy(2.0, 3.0, -0.4, 0.4, clearance=0.5)
    cfg = _cfg(
        cruise_speed=0.8,
        weight_obstacle=120.0,
        weight_progress=1.0,
        horizon_step_count=24,
    )
    mpc = DubinsPathFollowingMPC(
        vehicle_limits=_limits(max_speed=1.0, min_speed=0.05),
        config=cfg,
        occupancy=occ,
    )
    mpc.set_reference(straight_path)

    vehicle = DubinsVehicle(
        x=0.0,
        y=0.0,
        heading=0.0,
        max_speed=1.0,
        min_speed=0.05,
        max_turn_rate=1.2,
        max_acceleration=1.5,
        max_turn_rate_dot=2.0,
    )
    vehicle._speed = cfg.cruise_speed

    dt = cfg.dt
    steps = int(2.0 / dt)
    for _ in range(steps):
        result = mpc.step(
            vehicle.pose,
            speed=vehicle.speed,
            turn_rate=vehicle.turn_rate,
            dt=dt,
        )
        vehicle.step(result.speed_cmd, result.turn_rate_cmd, dt)

    assert vehicle.speed < 0.5 * cfg.cruise_speed


def test_mpc_avoids_lateral_obstacle(straight_path) -> None:
    # Wall parallel to the path, offset 0.5 m to the left.
    clearance = 0.4
    occ = RectOccupancy(1.0, 12.0, 0.5, 1.5, clearance=clearance)
    cfg = _cfg(
        cruise_speed=0.5,
        weight_obstacle=100.0,
        weight_contour=8.0,
    )
    mpc = DubinsPathFollowingMPC(
        vehicle_limits=_limits(),
        config=cfg,
        occupancy=occ,
    )
    mpc.set_reference(straight_path)

    vehicle = DubinsVehicle(
        x=0.0,
        y=0.15,  # slight bias toward the wall
        heading=0.0,
        max_speed=1.0,
        min_speed=0.05,
        max_turn_rate=1.2,
        max_acceleration=1.5,
        max_turn_rate_dot=2.0,
    )
    vehicle._speed = cfg.cruise_speed

    dt = cfg.dt
    steps = int(6.0 / dt)
    min_clearance = float("inf")
    for _ in range(steps):
        result = mpc.step(
            vehicle.pose,
            speed=vehicle.speed,
            turn_rate=vehicle.turn_rate,
            dt=dt,
        )
        vehicle.step(result.speed_cmd, result.turn_rate_cmd, dt)
        dist, _ = occ.nearest_obstacle(
            __import__("numpy").array(vehicle.pose[:2], dtype=float)
        )
        min_clearance = min(min_clearance, dist)

    assert min_clearance >= 0.8 * clearance


def test_mpc_respects_max_turn_rate(straight_path) -> None:
    max_turn_rate = 0.8
    # Gentle curve that requests turning within / near the limit.
    path = [(0.0, 0.0)]
    for i in range(1, 40):
        x = float(i) * 0.4
        y = 0.15 * math.sin(0.35 * x)
        path.append((x, y))

    cfg = _cfg(cruise_speed=0.45)
    mpc = DubinsPathFollowingMPC(
        vehicle_limits=_limits(max_turn_rate=max_turn_rate),
        config=cfg,
    )
    mpc.set_reference(path)

    vehicle = DubinsVehicle(
        x=0.0,
        y=0.0,
        heading=0.0,
        max_speed=1.0,
        min_speed=0.05,
        max_turn_rate=max_turn_rate,
        max_acceleration=1.5,
        max_turn_rate_dot=2.0,
    )
    vehicle._speed = cfg.cruise_speed

    dt = cfg.dt
    for _ in range(80):
        result = mpc.step(
            vehicle.pose,
            speed=vehicle.speed,
            turn_rate=vehicle.turn_rate,
            dt=dt,
        )
        assert abs(result.turn_rate_cmd) <= max_turn_rate * 1.01
        vehicle.step(result.speed_cmd, result.turn_rate_cmd, dt)
        assert abs(vehicle.turn_rate) <= max_turn_rate * 1.01


def test_mpc_solver_failure_safe_stop(straight_path) -> None:
    cfg = _cfg()
    mpc = DubinsPathFollowingMPC(vehicle_limits=_limits(), config=cfg)
    mpc.set_reference(straight_path)

    result = mpc.step(
        (float("nan"), 0.0, 0.0),
        speed=0.5,
        turn_rate=0.0,
        dt=cfg.dt,
    )
    assert result.solver_success is False
    # Deceleration relative to the corrupt call's speed argument.
    assert result.speed_cmd <= 0.5
    assert result.speed_cmd <= 0.5 - 0.5 * cfg.dt * 0.0 + 1e-9
    # Explicit safe-stop: a = -max_acceleration
    expected = max(0.05, 0.5 - _limits().max_acceleration * cfg.dt)
    assert abs(result.speed_cmd - expected) < 1e-9


def test_path_following_config_from_yaml() -> None:
    cfg = PathFollowingMPCConfig.create_from_config(cruise_speed=0.42)
    assert cfg.horizon_step_count == 20
    assert abs(cfg.dt - 0.05) < 1e-12
    assert abs(cfg.cruise_speed - 0.42) < 1e-12
    assert cfg.weight_obstacle > 0.0


def test_path_following_config_with_horizon_overrides() -> None:
    cfg = PathFollowingMPCConfig.create_from_config()
    longer = cfg.with_horizon_overrides(step_count=60, dt=0.05)
    assert longer.horizon_step_count == 60
    assert abs(longer.dt - 0.05) < 1e-12
    assert longer.weight_contour == cfg.weight_contour
    assert cfg.horizon_step_count == 20


def test_path_following_config_with_weight_overrides() -> None:
    cfg = PathFollowingMPCConfig.create_from_config()
    soft = cfg.with_weight_overrides(
        contour=1.5,
        heading=1.5,
        control=0.5,
        lag=10.0,
        contour_deadzone=8.0,
    )
    assert abs(soft.weight_contour - 1.5) < 1e-12
    assert abs(soft.weight_heading - 1.5) < 1e-12
    assert abs(soft.weight_control - 0.5) < 1e-12
    assert abs(soft.weight_lag - 10.0) < 1e-12
    assert abs(soft.contour_deadzone - 8.0) < 1e-12
    assert soft.horizon_step_count == cfg.horizon_step_count
    assert abs(cfg.weight_contour - 10.0) < 1e-12
    assert abs(cfg.weight_lag) < 1e-12
    assert abs(cfg.contour_deadzone) < 1e-12


def test_mpc_progress_does_not_reverse_when_heading_error_is_large() -> None:
    """Wider recovery arcs must not drive contouring progress backward.

    With the old ṡ = v cos(e_ψ) law, |e_ψ| > 90° reversed s and created the
    city A* junction limit cycle. Progress may stall, but not decrease.
    """
    path = [(float(i), 0.0) for i in range(30)]
    cfg = _cfg(cruise_speed=0.5, weight_contour=2.0, weight_heading=1.0)
    mpc = DubinsPathFollowingMPC(vehicle_limits=_limits(), config=cfg)
    mpc.set_reference(path)

    vehicle = DubinsVehicle(
        x=5.0,
        y=0.0,
        heading=math.pi,  # pointed opposite the path tangent
        max_speed=1.0,
        min_speed=0.05,
        max_turn_rate=1.2,
        max_acceleration=1.5,
        max_turn_rate_dot=2.0,
    )
    vehicle._speed = 0.5

    # Seed progress near mid-path so a reverse step would be visible.
    mpc._progress = 5.0
    progress_values = [5.0]
    dt = cfg.dt
    for _ in range(40):
        result = mpc.step(
            vehicle.pose,
            speed=vehicle.speed,
            turn_rate=vehicle.turn_rate,
            dt=dt,
        )
        progress_values.append(result.progress)
        vehicle.step(result.speed_cmd, result.turn_rate_cmd, dt)

    # Allow tiny numerical wobble, but no sustained regression.
    assert min(progress_values) >= 5.0 - 1e-3
    assert progress_values[-1] >= progress_values[0] - 1e-3


def test_mpc_progress_first_accepts_lateral_slip_to_advance() -> None:
    """Lag + deadzone prefer advancing s over hugging a sharp kink.

    On a right-angle path with limited turn rate, progress-first weights
    should keep arc-length increasing even while |e_lat| stays inside /
    near the free band — the opposite of stiff contouring that stalls.
    """
    path = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (4.0, 8.0)]
    cfg = _cfg(
        cruise_speed=0.8,
        weight_contour=2.0,
        weight_heading=1.0,
        weight_progress=0.5,
        weight_lag=12.0,
        contour_deadzone=0.8,
        weight_terminal=2.0,
        horizon_step_count=16,
    )
    # Tight turn rate so the corner cannot be tracked without slip.
    limits = _limits(max_turn_rate=0.35, max_acceleration=1.0)
    mpc = DubinsPathFollowingMPC(vehicle_limits=limits, config=cfg)
    mpc.set_reference(path)
    vehicle = DubinsVehicle(
        x=0.0,
        y=0.0,
        heading=0.0,
        max_speed=1.0,
        min_speed=0.05,
        max_turn_rate=0.35,
        max_acceleration=1.0,
        max_turn_rate_dot=2.0,
    )
    vehicle._speed = cfg.cruise_speed

    dt = cfg.dt
    progresses: list[float] = []
    max_abs_lat = 0.0
    for _ in range(160):
        result = mpc.step(
            vehicle.pose,
            speed=vehicle.speed,
            turn_rate=vehicle.turn_rate,
            dt=dt,
        )
        progresses.append(result.progress)
        max_abs_lat = max(max_abs_lat, abs(result.cross_track_error))
        vehicle.step(result.speed_cmd, result.turn_rate_cmd, dt)

    assert progresses[-1] > progresses[0] + 2.0
    # Must have used some of the free band / allowed slip near the corner.
    assert max_abs_lat > 0.15


def test_mpc_lane_aware_corner_stays_inside_road_budget() -> None:
    """City-scale L-kink: widen inside a small deadzone, not into walls.

    With corrected polyline κ (braking) + lane-aware weights, max |e_lat|
    must stay well below the 15 m road half-width while s still advances.
    """
    path = [
        (0.0, 0.0),
        (40.0, 0.0),
        (40.0, 40.0),
        (40.0, 80.0),
    ]
    road_half_width = 15.0
    deadzone = 2.5
    cfg = _cfg(
        cruise_speed=12.0,
        weight_contour=8.0,
        weight_heading=4.0,
        weight_progress=4.0,
        weight_lag=4.0,
        contour_deadzone=deadzone,
        weight_control=0.5,
        weight_obstacle=0.0,
        weight_terminal=8.0,
        horizon_step_count=72,
        dt=0.05,
        max_solver_iter_count=80,
    )
    limits = _limits(
        max_speed=16.0,
        min_speed=0.0,
        max_turn_rate=math.radians(40.0),
        max_acceleration=2.5,
        max_turn_rate_dot=math.radians(90.0),
    )
    mpc = DubinsPathFollowingMPC(vehicle_limits=limits, config=cfg)
    mpc.set_reference(path)
    vehicle = DubinsVehicle(
        x=0.0,
        y=0.0,
        heading=0.0,
        max_speed=limits.max_speed,
        min_speed=limits.min_speed,
        max_turn_rate=limits.max_turn_rate,
        max_acceleration=limits.max_acceleration,
        max_turn_rate_dot=limits.max_turn_rate_dot,
    )
    vehicle._speed = cfg.cruise_speed

    dt = cfg.dt
    progresses: list[float] = []
    max_abs_lat = 0.0
    for _ in range(int(20.0 / dt)):
        result = mpc.step(
            vehicle.pose,
            speed=vehicle.speed,
            turn_rate=vehicle.turn_rate,
            dt=dt,
        )
        progresses.append(result.progress)
        max_abs_lat = max(max_abs_lat, abs(result.cross_track_error))
        vehicle.step(result.speed_cmd, result.turn_rate_cmd, dt)
        if result.progress >= 70.0:
            break

    assert progresses[-1] > 50.0
    # Stay inside the navigable lane budget (clearance to buildings).
    assert max_abs_lat < road_half_width - 4.0
    # May use the free band, but must not treat half the road as free.
    assert max_abs_lat < 2.0 * deadzone + 3.0


def test_mpc_city_horizon_solves_dense_astar_style_kinks() -> None:
    """Same tracker as RRT*/SST must move on a dense A*-like polyline.

    v0.3.5 city release: purple (A*) path existed but IPOPT ``solve_failed``
    on every step (Dirac κ from ~1 m stubs + horizon 72) → permanent
    ``speed_cmd=0``.  RRT*/SST used the identical MPC factory and moved.
    """
    # Stair-step corridor with short stubs at each 90° corner (optimizer-like).
    path: list[tuple[float, float]] = [(0.0, 0.0)]
    x = 0.0
    y = 0.0
    for i in range(6):
        x += 12.0
        path.append((x, y))
        path.append((x + 1.2, y))  # short stub
        y += 12.0
        path.append((x + 1.2, y))
        x += 1.2
    path.append((x + 20.0, y))

    cfg = _cfg(
        cruise_speed=12.0,
        weight_contour=8.0,
        weight_heading=4.0,
        weight_progress=4.0,
        weight_lag=4.0,
        contour_deadzone=2.5,
        weight_control=0.5,
        weight_obstacle=0.0,
        weight_terminal=8.0,
        horizon_step_count=72,
        dt=0.05,
        max_solver_iter_count=80,
    )
    limits = _limits(
        max_speed=16.0,
        min_speed=0.0,
        max_turn_rate=math.radians(40.0),
        max_acceleration=2.5,
        max_turn_rate_dot=math.radians(90.0),
    )
    mpc = DubinsPathFollowingMPC(vehicle_limits=limits, config=cfg)
    mpc.set_reference(path)
    vehicle = DubinsVehicle(
        x=path[0][0],
        y=path[0][1],
        heading=0.0,
        max_speed=limits.max_speed,
        min_speed=limits.min_speed,
        max_turn_rate=limits.max_turn_rate,
        max_acceleration=limits.max_acceleration,
        max_turn_rate_dot=limits.max_turn_rate_dot,
    )
    # Match race start: vehicle begins at rest.
    vehicle._speed = 0.0

    dt = 0.1
    success_count = 0
    for _ in range(25):
        result = mpc.step(
            vehicle.pose,
            speed=vehicle.speed,
            turn_rate=vehicle.turn_rate,
            dt=dt,
        )
        if result.solver_success:
            success_count += 1
        vehicle.step(result.speed_cmd, result.turn_rate_cmd, dt)

    assert success_count >= 20
    assert vehicle.speed > 1.0
    assert math.hypot(vehicle.x - path[0][0], vehicle.y - path[0][1]) > 2.0
