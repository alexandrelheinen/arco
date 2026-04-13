"""Tests for PurePursuitController path tracking."""

from __future__ import annotations

import math

from arco.control.pure_pursuit import (
    PurePursuitController,
    _find_lookahead,
)

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


# ---------------------------------------------------------------------------
# _find_lookahead fallback regression test
# ---------------------------------------------------------------------------


def test_find_lookahead_does_not_return_goal_when_vehicle_is_far_off_track() -> (
    None
):
    """Fallback must target the next forward waypoint, not path[-1].

    Regression test for a bug where `_find_lookahead` returned ``path[-1]``
    (the goal) whenever the lookahead circle was too small to reach any path
    segment.  This caused the vehicle to steer directly to the goal whenever
    it drifted far off-track, bypassing all remaining waypoints.

    Scenario: straight path along y=0 from x=0 to x=100 (11 waypoints).
    Vehicle is 100 m above the path at (50, 100), lookahead radius = 5 m.
    The circle cannot intersect the path (which is 100 m below), so the
    fallback is triggered.  The correct behavior is to return the next
    forward waypoint (60, 0), not the goal (100, 0).
    """
    path: list[tuple[float, float]] = [
        (float(x), 0.0) for x in range(0, 101, 10)
    ]
    start_idx = 5  # closest waypoint: (50, 0)

    result = _find_lookahead(50.0, 100.0, path, start_idx, lookahead=5.0)

    goal = path[-1]  # (100, 0)
    assert result[0] != goal[0] or result[1] != goal[1], (
        f"_find_lookahead returned path[-1]={goal} while vehicle was 100 m "
        "off-track — lookahead fell back to goal instead of next waypoint."
    )
    # The fallback target must be close to the current path position, not at the far end.
    assert result[0] < 70.0, (
        f"Fallback target x={result[0]:.1f} is too far ahead (expected ≤60, "
        "i.e. the next waypoint, not the goal)."
    )


# ---------------------------------------------------------------------------
# Curvature attribute tests
# ---------------------------------------------------------------------------


def test_curvature_is_float() -> None:
    ctrl = PurePursuitController(lookahead_distance=1.0)
    ctrl.track((0.0, 0.0, 0.0), STRAIGHT_PATH, speed=1.0)
    assert isinstance(ctrl.curvature, float)


def test_curvature_zero_on_straight_aligned_path() -> None:
    """Vehicle on path and aligned with it → curvature ≈ 0."""
    ctrl = PurePursuitController(lookahead_distance=2.0)
    ctrl.track((0.0, 0.0, 0.0), STRAIGHT_PATH, speed=1.0)
    assert abs(ctrl.curvature) < 0.05


def test_curvature_sign_matches_turn_direction() -> None:
    """Vehicle above path (positive y) steers right → negative curvature."""
    ctrl = PurePursuitController(lookahead_distance=2.0)
    ctrl.track((0.0, 1.0, 0.0), STRAIGHT_PATH, speed=1.0)
    assert ctrl.curvature < 0.0


def test_curvature_magnitude_consistent_with_turn_rate() -> None:
    """turn_rate_cmd must equal speed * curvature."""
    ctrl = PurePursuitController(lookahead_distance=2.0)
    speed = 2.0
    _, turn_rate = ctrl.track((0.0, 1.0, 0.0), STRAIGHT_PATH, speed=speed)
    assert math.isclose(turn_rate, speed * ctrl.curvature, rel_tol=1e-9)


def test_control_interface_proportional() -> None:
    ctrl = PurePursuitController(lookahead_distance=1.0)
    assert ctrl.control(0.5, 2.0) == 1.5


# ---------------------------------------------------------------------------
# Lookahead between widely-spaced waypoints
# ---------------------------------------------------------------------------


def test_lookahead_found_on_enclosing_segment_not_path_end() -> None:
    """Lookahead must use the segment enclosing the vehicle, not path[-1].

    When the vehicle is past the midpoint between two waypoints, the
    globally closest waypoint is the one *ahead*.  The lookahead should
    still intersect the segment that the vehicle is currently traversing
    (the one ending at that ahead waypoint), not fall back to path[-1].

    Regression test for the bug where ``_find_lookahead`` started scanning
    from the closest waypoint index and therefore skipped the enclosing
    segment entirely.
    """
    # Path with 50 m spacing — much larger than the 10 m lookahead
    path = [
        (0.0, 0.0),
        (50.0, 0.0),
        (100.0, 0.0),
        (150.0, 0.0),
    ]
    # Vehicle at x=30, y=2: past the mid-point of segment [0]→[1], 2 m above
    # the path.  The closest waypoint is [1]=(50,0) at ~20 m; without the fix
    # _find_lookahead would scan from index 1 onward, find no intersection
    # (those segments start at x≥50, all > 10 m away), and return path[-1].
    pose = (30.0, 2.0, 0.0)
    ctrl = PurePursuitController(lookahead_distance=10.0)
    _, turn_rate = ctrl.track(pose, path, speed=1.0)

    # With the fix the lookahead lands on segment [0]→[1] at ≈(40, 0),
    # requiring a noticeable right-turn correction for the 2 m offset.
    # With the bug the lookahead is path[-1]=(150, 0), far ahead along
    # the same heading, yielding an almost-zero turn rate.
    assert turn_rate < -0.02, (
        f"Turn rate {turn_rate:.4f} rad/s should be noticeably negative "
        f"(target on enclosing segment); near-zero indicates lookahead "
        f"fell back to path[-1] instead."
    )
