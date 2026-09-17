"""The geometry, tolerance and generator surface of ``arco._arco``.

These names come straight from ``arco-core`` and the Python library
never re-exported them, so nothing else in the suite reaches them. They
are the cheapest place to prove that the boundary converts a value
correctly and rejects a value that has no meaning, which is what
``FR-SAFE-01`` asks of every exported function.

Every numeric comparison below states its own tolerance in the unit of
the quantity, per ``.guidelines/languages/py.md``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from arco._arco import (
    ANGLE_TOLERANCE,
    PCG64,
    POSITION_TOLERANCE,
    RELATIVE_TOLERANCE,
    TIME_TOLERANCE,
    Pose,
    angle_difference,
    angles_close,
    euclidean_distance,
    is_close,
    manhattan_distance,
    points_close,
    positions_close,
    wrap_angle,
)

# Comparison tolerances for the assertions below, in meters and radians.
METERS = 1e-12
RADIANS = 1e-12


# ---------------------------------------------------------------------
# Pose
# ---------------------------------------------------------------------


def test_a_pose_keeps_the_position_it_was_built_with():
    pose = Pose(1.5, -2.25, 0.5)
    assert pose.x == pytest.approx(1.5, abs=METERS)
    assert pose.y == pytest.approx(-2.25, abs=METERS)
    assert pose.heading == pytest.approx(0.5, abs=RADIANS)


def test_a_pose_wraps_its_heading_into_the_half_open_turn():
    pose = Pose(0.0, 0.0, 3.0 * math.pi)
    assert -math.pi <= pose.heading < math.pi
    assert pose.heading == pytest.approx(-math.pi, abs=RADIANS)


def test_a_pose_rejects_a_position_that_is_not_a_real_number():
    with pytest.raises(ValueError, match="x is NaN, which is not finite"):
        Pose(float("nan"), 0.0, 0.0)
    with pytest.raises(ValueError, match="y is inf, which is not finite"):
        Pose(0.0, float("inf"), 0.0)


def test_a_pose_rejects_a_heading_that_is_not_a_real_number():
    with pytest.raises(ValueError, match="angle is NaN, which is not finite"):
        Pose(0.0, 0.0, float("nan"))


def test_the_distance_between_two_poses_ignores_their_headings():
    origin = Pose(0.0, 0.0, 0.0)
    corner = Pose(3.0, 4.0, 2.0)
    assert origin.distance_to(corner) == pytest.approx(5.0, abs=METERS)
    assert corner.distance_to(origin) == pytest.approx(5.0, abs=METERS)


def test_a_heading_difference_stays_small_across_the_branch_cut():
    just_below = Pose(0.0, 0.0, math.pi - 0.1)
    just_above = Pose(0.0, 0.0, -math.pi + 0.1)
    difference = just_above.heading_difference(just_below)
    assert difference == pytest.approx(0.2, abs=RADIANS)


def test_a_pose_prints_its_three_components():
    assert repr(Pose(1.0, 2.0, 0.0)) == "Pose(x=1, y=2, heading=0)"


# ---------------------------------------------------------------------
# The distance helpers
# ---------------------------------------------------------------------


def test_the_euclidean_distance_is_the_straight_line_between_points():
    assert euclidean_distance([0.0, 0.0], [3.0, 4.0]) == pytest.approx(
        5.0, abs=METERS
    )


def test_the_euclidean_distance_reads_a_numpy_array_as_a_point():
    left = np.array([0.0, 0.0, 0.0])
    right = np.array([1.0, 2.0, 2.0])
    assert euclidean_distance(left, right) == pytest.approx(3.0, abs=METERS)


def test_the_manhattan_distance_sums_the_per_axis_differences():
    assert manhattan_distance([0.0, 0.0], [3.0, -4.0]) == pytest.approx(
        7.0, abs=METERS
    )


@pytest.mark.parametrize(
    "measure", [euclidean_distance, manhattan_distance, points_close]
)
def test_a_distance_rejects_two_points_of_different_length(measure):
    with pytest.raises(ValueError, match=r"point has dimension 1, expected 2"):
        measure([1.0, 2.0], [1.0])


@pytest.mark.parametrize(
    "measure", [euclidean_distance, manhattan_distance, points_close]
)
def test_a_distance_rejects_a_coordinate_that_is_not_finite(measure):
    with pytest.raises(ValueError, match="point is NaN, which is not finite"):
        measure([float("nan"), 0.0], [1.0, 1.0])


def test_a_distance_rejects_an_argument_that_is_not_a_sequence():
    with pytest.raises(TypeError):
        euclidean_distance("here", [1.0, 2.0])


def test_two_points_are_close_to_themselves_and_not_to_a_far_one():
    assert points_close([1.0, 2.0], [1.0, 2.0])
    assert not points_close([1.0, 2.0], [1.0, 3.0])


# ---------------------------------------------------------------------
# The tolerance helpers
# ---------------------------------------------------------------------


def test_a_comparison_governed_by_the_absolute_tolerance_near_zero():
    assert is_close(0.0, 1e-9, absolute=1e-6, relative=0.0)
    assert not is_close(0.0, 1e-3, absolute=1e-6, relative=0.0)


def test_a_comparison_governed_by_the_relative_tolerance_away_from_zero():
    assert is_close(1e6, 1e6 + 1.0, absolute=0.0, relative=1e-3)
    assert not is_close(1e6, 1e6 + 1.0, absolute=0.0, relative=1e-9)


def test_two_positions_agree_within_the_position_tolerance():
    assert positions_close(1.0, 1.0 + POSITION_TOLERANCE / 10.0)
    assert not positions_close(1.0, 1.0 + POSITION_TOLERANCE * 1e4)


def test_two_angles_agree_across_the_branch_cut():
    assert angles_close(math.pi, -math.pi)
    assert not angles_close(0.0, 1.0)


def test_an_angle_agrees_with_nothing_once_it_is_not_a_real_number():
    assert not angles_close(float("nan"), float("nan"))
    assert not angles_close(float("inf"), float("inf"))


def test_wrapping_an_angle_lands_inside_the_half_open_turn():
    assert wrap_angle(3.0 * math.pi) == pytest.approx(-math.pi, abs=RADIANS)
    assert wrap_angle(0.25) == pytest.approx(0.25, abs=RADIANS)


def test_wrapping_rejects_an_angle_with_no_representative():
    with pytest.raises(ValueError, match="angle is NaN, which is not finite"):
        wrap_angle(float("nan"))
    with pytest.raises(ValueError, match="angle is inf, which is not finite"):
        wrap_angle(float("inf"))


def test_an_angle_difference_is_signed_and_wrapped():
    assert angle_difference(0.1, -0.1) == pytest.approx(0.2, abs=RADIANS)
    assert angle_difference(-math.pi + 0.1, math.pi - 0.1) == pytest.approx(
        0.2, abs=RADIANS
    )


def test_an_angle_difference_rejects_an_argument_that_is_not_finite():
    with pytest.raises(ValueError, match="angle is inf, which is not finite"):
        angle_difference(float("inf"), 0.0)


def test_every_published_tolerance_is_a_small_positive_number():
    published = (
        POSITION_TOLERANCE,
        ANGLE_TOLERANCE,
        TIME_TOLERANCE,
        RELATIVE_TOLERANCE,
    )
    for tolerance in published:
        assert isinstance(tolerance, float)
        assert 0.0 < tolerance < 1e-3


# ---------------------------------------------------------------------
# PCG64
# ---------------------------------------------------------------------


def test_the_generator_draws_the_stream_numpy_draws_from_the_same_seed():
    """FR-RNG-02: numpy is the oracle, not a value recorded from Rust."""
    seed = 20260917
    drawn = [PCG64(seed).next_f64() for _ in range(1)]
    expected = np.random.default_rng(seed).random(1).tolist()
    assert drawn == pytest.approx(expected, abs=0.0, rel=0.0)


def test_a_sequence_of_draws_follows_the_numpy_sequence():
    seed = 7
    generator = PCG64(seed)
    drawn = [generator.next_f64() for _ in range(8)]
    expected = np.random.default_rng(seed).random(8).tolist()
    assert drawn == pytest.approx(expected, abs=0.0, rel=0.0)


def test_every_double_drawn_lies_in_the_unit_interval():
    generator = PCG64(99)
    for _ in range(64):
        value = generator.next_f64()
        assert 0.0 <= value < 1.0


def test_a_raw_draw_fits_in_sixty_four_bits_and_advances_the_state():
    generator = PCG64(3)
    before = generator.state
    value = generator.next_u64()
    assert 0 <= value < 2**64
    assert generator.state != before


def test_two_generators_seeded_alike_agree_draw_for_draw():
    first = PCG64(41)
    second = PCG64(41)
    assert first.state == second.state
    assert first.increment == second.increment
    assert [first.next_u64() for _ in range(4)] == [
        second.next_u64() for _ in range(4)
    ]


def test_two_generators_seeded_differently_start_apart():
    assert PCG64(1).state != PCG64(2).state


def test_a_generator_prints_its_state_and_its_stream():
    generator = PCG64(5)
    printed = repr(generator)
    assert printed == (
        f"PCG64(state={generator.state}, " f"increment={generator.increment})"
    )


def test_a_seed_outside_the_sixty_four_bit_range_is_refused():
    with pytest.raises(OverflowError):
        PCG64(-1)


def test_the_module_reports_the_math_module_agrees_on_wrapping():
    """A second oracle for wrap_angle, independent of the binding."""
    for angle in (-7.0, -1.0, 0.0, 1.0, 7.0, 100.0):
        wrapped = wrap_angle(angle)
        assert math.isclose(
            math.sin(wrapped),
            math.sin(angle),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        assert math.isclose(
            math.cos(wrapped),
            math.cos(angle),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
