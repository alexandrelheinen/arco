"""Tests for MPCTrackingLoop metrics schema and factory wiring."""

from __future__ import annotations

import pytest

casadi = pytest.importorskip("casadi")

from arco.control.mpc import (  # noqa: E402
    DubinsPathFollowingMPC,
    DubinsVehicleLimits,
    MPCTrackingLoop,
    PathFollowingMPCConfig,
)
from arco.guidance.vehicle import DubinsVehicle  # noqa: E402
from arco.simulator.sim.tracking import (  # noqa: E402
    VehicleConfig,
    build_vehicle_mpc_sim,
)


def test_mpc_tracking_loop_metrics_schema() -> None:
    path = [(float(i), 0.0) for i in range(20)]
    vehicle = DubinsVehicle(
        x=0.0,
        y=0.0,
        heading=0.0,
        max_speed=1.0,
        min_speed=0.05,
        max_turn_rate=1.0,
        max_acceleration=1.5,
        max_turn_rate_dot=2.0,
    )
    vehicle._speed = 0.4
    tracker = DubinsPathFollowingMPC(
        vehicle_limits=DubinsVehicleLimits(
            max_speed=1.0,
            min_speed=0.05,
            max_turn_rate=1.0,
            max_acceleration=1.5,
            max_turn_rate_dot=2.0,
        ),
        config=PathFollowingMPCConfig(cruise_speed=0.4, dt=0.05),
    )
    loop = MPCTrackingLoop(vehicle, tracker, cruise_speed=0.4)
    metrics = loop.step(path, dt=0.05)
    for key in (
        "cross_track_error",
        "heading_error",
        "pose",
        "speed",
        "turn_rate",
        "curvature",
        "repulsion_turn_rate",
        "mpc_solver_success",
        "mpc_solve_time_s",
        "mpc_cost",
        "mpc_progress",
        "mpc_predicted_clearance_min",
        "mpc_predicted_xy",
    ):
        assert key in metrics
    assert isinstance(metrics["mpc_predicted_xy"], list)
    assert len(metrics["mpc_predicted_xy"]) >= 2
    x0, y0 = metrics["mpc_predicted_xy"][0]
    assert abs(x0 - 0.0) < 1e-6
    assert abs(y0 - 0.0) < 1e-6


def test_build_vehicle_mpc_sim_factory() -> None:
    path = [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)]
    cfg = VehicleConfig(
        max_speed=1.0,
        min_speed=0.05,
        cruise_speed=0.4,
        lookahead_distance=1.0,
        goal_radius=0.2,
        max_turn_rate=1.0,
        max_acceleration=1.5,
        max_turn_rate_dot=2.0,
    )
    mpc_cfg = PathFollowingMPCConfig.create_from_config()
    vehicle, loop = build_vehicle_mpc_sim(path, cfg, mpc_cfg)
    assert isinstance(loop, MPCTrackingLoop)
    assert abs(vehicle.x - 0.0) < 1e-12
    metrics = loop.step(path, dt=0.05)
    assert "mpc_solver_success" in metrics
