"""Tests for DubinsVehicle kinematic model."""

from __future__ import annotations

import math

from arco.guidance.vehicle import DubinsVehicle


def test_initial_state() -> None:
    vehicle = DubinsVehicle(x=1.0, y=2.0, heading=0.5)
    assert vehicle.x == 1.0
    assert vehicle.y == 2.0
    assert vehicle.heading == 0.5
    assert vehicle.speed == 0.0
    assert vehicle.turn_rate == 0.0


def test_pose_property() -> None:
    vehicle = DubinsVehicle(x=1.0, y=2.0, heading=0.5)
    assert vehicle.pose == (1.0, 2.0, 0.5)


def test_step_returns_pose_tuple() -> None:
    vehicle = DubinsVehicle()
    pose = vehicle.step(1.0, 0.0, 0.1)
    assert isinstance(pose, tuple)
    assert len(pose) == 3


def test_speed_saturation() -> None:
    """Speed command exceeding max_speed must be clamped."""
    vehicle = DubinsVehicle(max_speed=2.0, max_acceleration=1000.0)
    vehicle.step(100.0, 0.0, 0.1)
    assert vehicle.speed <= 2.0 + 1e-9


def test_min_speed_saturation() -> None:
    """Speed command below min_speed must be clamped (no reverse by default)."""
    vehicle = DubinsVehicle(min_speed=0.0, max_acceleration=1000.0)
    vehicle.step(-10.0, 0.0, 0.1)
    assert vehicle.speed >= 0.0 - 1e-9


def test_turn_rate_saturation() -> None:
    """Turn rate command exceeding max_turn_rate must be clamped."""
    vehicle = DubinsVehicle(max_turn_rate=1.0, max_turn_rate_dot=1000.0)
    vehicle.step(1.0, 100.0, 0.1)
    assert abs(vehicle.turn_rate) <= 1.0 + 1e-9


def test_acceleration_filtering() -> None:
    """Speed changes must be rate-limited by max_acceleration."""
    vehicle = DubinsVehicle(max_speed=100.0, max_acceleration=1.0)
    vehicle.step(100.0, 0.0, 0.1)
    # After one step: speed ≤ 0 + 1.0 * 0.1 = 0.1
    assert vehicle.speed <= 1.0 * 0.1 + 1e-9


def test_turn_rate_filtering() -> None:
    """Turn rate changes must be rate-limited by max_turn_rate_dot."""
    vehicle = DubinsVehicle(max_turn_rate=100.0, max_turn_rate_dot=1.0)
    vehicle.step(1.0, 100.0, 0.1)
    # After one step: |turn_rate| ≤ 1.0 * 0.1 = 0.1
    assert abs(vehicle.turn_rate) <= 1.0 * 0.1 + 1e-9


def test_straight_line_motion() -> None:
    """Vehicle with zero turn rate and east heading must move along x-axis."""
    vehicle = DubinsVehicle(heading=0.0, max_acceleration=1000.0)
    vehicle.step(1.0, 0.0, 1.0)
    assert math.isclose(vehicle.y, 0.0, abs_tol=1e-9)
    assert vehicle.x > 0.0


def test_heading_normalized() -> None:
    """Heading must remain within [−π, π] after many turns."""
    vehicle = DubinsVehicle(
        heading=3.0, max_turn_rate=1000.0, max_turn_rate_dot=1000.0
    )
    for _ in range(100):
        vehicle.step(1.0, 5.0, 0.1)
    assert -math.pi <= vehicle.heading <= math.pi


def test_reset_restores_state() -> None:
    """Reset must restore the given pose and zero speed/turn-rate."""
    vehicle = DubinsVehicle(x=5.0, y=5.0, heading=1.0)
    vehicle.step(2.0, 1.0, 0.5)
    vehicle.reset(0.0, 0.0, 0.0)
    assert vehicle.pose == (0.0, 0.0, 0.0)
    assert vehicle.speed == 0.0
    assert vehicle.turn_rate == 0.0


def test_command_bounds_over_many_steps() -> None:
    """Speed and turn rate must stay within bounds across a long simulation."""
    vehicle = DubinsVehicle(
        max_speed=3.0,
        min_speed=0.0,
        max_turn_rate=1.5,
        max_acceleration=1000.0,
    )
    for _ in range(200):
        vehicle.step(10.0, 10.0, 0.05)
        assert vehicle.speed <= 3.0 + 1e-9
        assert vehicle.speed >= 0.0 - 1e-9
        assert abs(vehicle.turn_rate) <= 1.5 + 1e-9
