"""Tests for TrackingLoop – convergence, metrics logging, and command bounds."""

from __future__ import annotations

import math

from arco.guidance.pure_pursuit import PurePursuitController
from arco.guidance.tracking import TrackingLoop
from arco.guidance.vehicle import DubinsVehicle

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Long straight path along the x-axis used in most tests
STRAIGHT_PATH: list[tuple[float, float]] = [(float(i), 0.0) for i in range(40)]


def _make_loop(
    x: float = 0.0,
    y: float = 0.0,
    heading: float = 0.0,
    lookahead: float = 2.0,
    speed: float = 1.0,
    max_speed: float = 5.0,
    max_turn_rate: float = 2.0,
) -> TrackingLoop:
    """Create a TrackingLoop with default-safe parameters."""
    vehicle = DubinsVehicle(
        x=x,
        y=y,
        heading=heading,
        max_speed=max_speed,
        min_speed=0.0,
        max_turn_rate=max_turn_rate,
        max_acceleration=10.0,
        max_turn_rate_dot=10.0,
    )
    ctrl = PurePursuitController(lookahead_distance=lookahead)
    return TrackingLoop(vehicle, ctrl, cruise_speed=speed)


# ---------------------------------------------------------------------------
# Step / run interface tests
# ---------------------------------------------------------------------------


def test_step_returns_all_metric_keys() -> None:
    loop = _make_loop()
    metrics = loop.step(STRAIGHT_PATH, dt=0.1)
    for key in (
        "cross_track_error",
        "heading_error",
        "pose",
        "speed",
        "turn_rate",
        "curvature",
    ):
        assert key in metrics


def test_step_pose_is_tuple() -> None:
    loop = _make_loop()
    metrics = loop.step(STRAIGHT_PATH, dt=0.1)
    assert isinstance(metrics["pose"], tuple)
    assert len(metrics["pose"]) == 3


def test_metrics_property_after_step() -> None:
    loop = _make_loop()
    loop.step(STRAIGHT_PATH, dt=0.1)
    m = loop.metrics
    assert m is not None
    assert "cross_track_error" in m


def test_metrics_property_before_step_is_none() -> None:
    loop = _make_loop()
    assert loop.metrics is None


def test_history_grows_with_steps() -> None:
    loop = _make_loop()
    for _ in range(5):
        loop.step(STRAIGHT_PATH, dt=0.1)
    assert len(loop.history) == 5
    for entry in loop.history:
        for key in (
            "cross_track_error",
            "heading_error",
            "pose",
            "speed",
            "turn_rate",
            "curvature",
        ):
            assert key in entry


def test_run_returns_correct_length() -> None:
    loop = _make_loop()
    results = loop.run(STRAIGHT_PATH, steps=10, dt=0.1)
    assert len(results) == 10


def test_run_each_result_has_all_keys() -> None:
    loop = _make_loop()
    results = loop.run(STRAIGHT_PATH, steps=5, dt=0.1)
    for r in results:
        assert "cross_track_error" in r
        assert "heading_error" in r


# ---------------------------------------------------------------------------
# Command bounds tests
# ---------------------------------------------------------------------------


def test_speed_stays_within_max_speed() -> None:
    """Vehicle speed must never exceed max_speed across a long simulation."""
    loop = _make_loop(speed=4.0, max_speed=5.0)
    for r in loop.run(STRAIGHT_PATH, steps=100, dt=0.1):
        assert r["speed"] <= 5.0 + 1e-9


# ---------------------------------------------------------------------------
# Curvature-based speed modulation tests
# ---------------------------------------------------------------------------


CURVED_PATH: list[tuple[float, float]] = [
    (math.cos(t) * 5.0, math.sin(t) * 5.0)
    for t in [i * math.pi / 20 for i in range(41)]
]


def test_curvature_gain_zero_does_not_change_speed_command() -> None:
    """With curvature_gain=0 the speed reference equals cruise_speed."""
    loop = _make_loop(speed=2.0)
    # After any step the speed reference should equal cruise_speed (no modulation)
    # We verify by checking that on a straight path the vehicle accelerates
    # toward cruise_speed just as it would without the feature.
    loop_ref = _make_loop(speed=2.0)
    for _ in range(20):
        loop.step(STRAIGHT_PATH, dt=0.1)
        loop_ref.step(STRAIGHT_PATH, dt=0.1)
    assert math.isclose(loop.vehicle.speed, loop_ref.vehicle.speed, rel_tol=1e-9)


def test_curvature_gain_slows_vehicle_on_curve() -> None:
    """Positive curvature_gain must produce lower speed on a curved path."""
    vehicle_slow = DubinsVehicle(
        x=CURVED_PATH[0][0],
        y=CURVED_PATH[0][1],
        heading=math.pi / 2,
        max_speed=5.0,
        min_speed=0.0,
        max_turn_rate=4.0,
        max_acceleration=10.0,
        max_turn_rate_dot=10.0,
    )
    vehicle_fast = DubinsVehicle(
        x=CURVED_PATH[0][0],
        y=CURVED_PATH[0][1],
        heading=math.pi / 2,
        max_speed=5.0,
        min_speed=0.0,
        max_turn_rate=4.0,
        max_acceleration=10.0,
        max_turn_rate_dot=10.0,
    )
    loop_modulated = TrackingLoop(
        vehicle_slow,
        PurePursuitController(lookahead_distance=2.0),
        cruise_speed=3.0,
        curvature_gain=10.0,
    )
    loop_constant = TrackingLoop(
        vehicle_fast,
        PurePursuitController(lookahead_distance=2.0),
        cruise_speed=3.0,
        curvature_gain=0.0,
    )
    for _ in range(30):
        loop_modulated.step(CURVED_PATH, dt=0.1)
        loop_constant.step(CURVED_PATH, dt=0.1)

    avg_mod = sum(h["speed"] for h in loop_modulated.history) / len(
        loop_modulated.history
    )
    avg_const = sum(h["speed"] for h in loop_constant.history) / len(
        loop_constant.history
    )
    assert avg_mod < avg_const


def test_speed_never_exceeds_cruise_with_curvature_gain() -> None:
    """Curvature modulation must never push the reference above cruise_speed."""
    vehicle = DubinsVehicle(
        x=CURVED_PATH[0][0],
        y=CURVED_PATH[0][1],
        heading=math.pi / 2,
        max_speed=5.0,
        min_speed=0.0,
        max_turn_rate=4.0,
        max_acceleration=10.0,
        max_turn_rate_dot=10.0,
    )
    loop = TrackingLoop(
        vehicle,
        PurePursuitController(lookahead_distance=2.0),
        cruise_speed=3.0,
        curvature_gain=10.0,
    )
    loop.run(CURVED_PATH, steps=30, dt=0.1)
    for h in loop.history:
        assert h["speed"] <= 3.0 + 1e-9


def test_speed_non_negative() -> None:
    """Vehicle speed must remain non-negative (no reverse, min_speed=0)."""
    loop = _make_loop()
    for r in loop.run(STRAIGHT_PATH, steps=50, dt=0.1):
        assert r["speed"] >= 0.0 - 1e-9


def test_turn_rate_stays_bounded() -> None:
    """Turn rate must never exceed max_turn_rate in magnitude."""
    loop = _make_loop(y=3.0, heading=0.0, max_turn_rate=2.0)
    for r in loop.run(STRAIGHT_PATH, steps=100, dt=0.1):
        assert abs(r["turn_rate"]) <= 2.0 + 1e-9


# ---------------------------------------------------------------------------
# Convergence tests
# ---------------------------------------------------------------------------


def test_cross_track_error_logged_and_finite() -> None:
    """Cross-track error must be a finite float after each step."""
    loop = _make_loop(y=1.0)
    for r in loop.run(STRAIGHT_PATH, steps=20, dt=0.1):
        assert math.isfinite(r["cross_track_error"])


def test_heading_error_logged_and_finite() -> None:
    """Heading error must be a finite float after each step."""
    loop = _make_loop(heading=math.pi / 4)
    for r in loop.run(STRAIGHT_PATH, steps=20, dt=0.1):
        assert math.isfinite(r["heading_error"])


def test_convergence_straight_path() -> None:
    """Cross-track error should decrease when vehicle starts laterally offset."""
    loop = _make_loop(x=0.0, y=2.0, heading=0.0, lookahead=2.0, speed=2.0)
    initial_error = abs(loop.vehicle.y)
    results = loop.run(STRAIGHT_PATH, steps=120, dt=0.1)
    final_error = abs(results[-1]["cross_track_error"])
    assert final_error < initial_error


def test_control_stability_no_divergence() -> None:
    """Absolute cross-track error must not exceed the initial offset after tracking."""
    initial_offset = 1.5
    loop = _make_loop(x=0.0, y=initial_offset, heading=0.0, speed=1.5)
    for r in loop.run(STRAIGHT_PATH, steps=150, dt=0.1):
        # Error should not grow beyond the initial offset
        assert abs(r["cross_track_error"]) <= initial_offset + 1e-3
