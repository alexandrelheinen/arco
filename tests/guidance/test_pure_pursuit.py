"""Tests for PurePursuitController path tracking."""

from __future__ import annotations

import math

from arco.guidance.pure_pursuit import PurePursuitController

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Straight path along the x-axis
STRAIGHT_PATH: list[tuple[float, float]] = [(float(i), 0.0) for i in range(20)]


# ---------------------------------------------------------------------------
# track() interface tests
# ---------------------------------------------------------------------------


def test_track_returns_two_floats() -> None:
    ctrl = PurePursuitController(lookahead_distance=1.0)
    result = ctrl.track((0.0, 0.0, 0.0), STRAIGHT_PATH)
    assert isinstance(result, tuple)
    assert len(result) == 2
    speed_cmd, turn_rate_cmd = result
    assert isinstance(speed_cmd, float)
    assert isinstance(turn_rate_cmd, float)


def test_track_speed_passed_through() -> None:
    ctrl = PurePursuitController(lookahead_distance=1.0)
    speed_cmd, _ = ctrl.track((0.0, 0.0, 0.0), STRAIGHT_PATH, speed=2.5)
    assert speed_cmd == 2.5


def test_track_aligned_with_path_near_zero_turn() -> None:
    """Vehicle on path and aligned with it should command near-zero turn rate."""
    ctrl = PurePursuitController(lookahead_distance=2.0)
    _, turn_rate = ctrl.track((0.0, 0.0, 0.0), STRAIGHT_PATH, speed=1.0)
    assert abs(turn_rate) < 0.1


def test_track_single_waypoint_no_crash() -> None:
    """Single-point path (< 2 waypoints) must not raise and returns zero turn."""
    ctrl = PurePursuitController(lookahead_distance=1.0)
    speed_cmd, turn_rate = ctrl.track((0.0, 0.0, 0.0), [(0.0, 0.0)], speed=1.5)
    assert turn_rate == 0.0
    assert speed_cmd == 1.5


# ---------------------------------------------------------------------------
# Error metric logging tests
# ---------------------------------------------------------------------------


def test_cross_track_error_is_float() -> None:
    ctrl = PurePursuitController(lookahead_distance=1.0)
    ctrl.track((0.0, 1.0, 0.0), STRAIGHT_PATH, speed=1.0)
    assert isinstance(ctrl.cross_track_error, float)


def test_heading_error_is_float() -> None:
    ctrl = PurePursuitController(lookahead_distance=1.0)
    ctrl.track((0.0, 0.0, math.pi / 4), STRAIGHT_PATH, speed=1.0)
    assert isinstance(ctrl.heading_error, float)


def test_cross_track_error_positive_when_left_of_path() -> None:
    """Vehicle above the x-axis path → positive (left) cross-track error."""
    ctrl = PurePursuitController(lookahead_distance=2.0)
    ctrl.track((5.0, 1.0, 0.0), STRAIGHT_PATH, speed=1.0)
    assert ctrl.cross_track_error > 0.0


def test_cross_track_error_negative_when_right_of_path() -> None:
    """Vehicle below the x-axis path → negative (right) cross-track error."""
    ctrl = PurePursuitController(lookahead_distance=2.0)
    ctrl.track((5.0, -1.0, 0.0), STRAIGHT_PATH, speed=1.0)
    assert ctrl.cross_track_error < 0.0


def test_heading_error_zero_when_aligned() -> None:
    """Vehicle heading east and path heading east → heading error ≈ 0."""
    ctrl = PurePursuitController(lookahead_distance=2.0)
    ctrl.track((0.0, 0.0, 0.0), STRAIGHT_PATH, speed=1.0)
    assert math.isclose(ctrl.heading_error, 0.0, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# Steering direction tests
# ---------------------------------------------------------------------------


def test_turn_toward_path_when_offset_above() -> None:
    """Vehicle above (positive y) straight path should turn right (ω < 0)."""
    ctrl = PurePursuitController(lookahead_distance=2.0)
    _, turn_rate = ctrl.track((0.0, 1.0, 0.0), STRAIGHT_PATH, speed=1.0)
    assert turn_rate < 0.0


def test_turn_toward_path_when_offset_below() -> None:
    """Vehicle below (negative y) straight path should turn left (ω > 0)."""
    ctrl = PurePursuitController(lookahead_distance=2.0)
    _, turn_rate = ctrl.track((0.0, -1.0, 0.0), STRAIGHT_PATH, speed=1.0)
    assert turn_rate > 0.0


# ---------------------------------------------------------------------------
# Backward-compatible control() interface
# ---------------------------------------------------------------------------


def test_control_interface_returns_float() -> None:
    ctrl = PurePursuitController(lookahead_distance=1.0)
    cmd = ctrl.control(0.0, 1.0)
    assert isinstance(cmd, float)


def test_control_interface_proportional() -> None:
    ctrl = PurePursuitController(lookahead_distance=1.0)
    assert ctrl.control(0.5, 2.0) == 1.5
