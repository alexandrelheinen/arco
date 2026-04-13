"""Tests for arco.control.actuator — ActuatorArray."""

from __future__ import annotations

import math

import numpy as np
import pytest

from arco.control.actuator import ActuatorArray
from arco.control.rigid_body import CircleBody, SquareBody

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def square() -> SquareBody:
    return SquareBody(mass=5.0, side_length=0.5)


@pytest.fixture()
def circle() -> CircleBody:
    return CircleBody(mass=5.0, radius=0.3)


@pytest.fixture()
def array4() -> ActuatorArray:
    return ActuatorArray(actuator_count=4, standoff=0.05)


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestActuatorArrayInit:
    def test_default_count(self) -> None:
        a = ActuatorArray()
        assert a.actuator_count == 4

    def test_custom_count(self) -> None:
        a = ActuatorArray(actuator_count=6)
        assert a.actuator_count == 6

    def test_angles_length(self, array4: ActuatorArray) -> None:
        assert len(array4.angles) == 4

    def test_count_less_than_3_raises(self) -> None:
        with pytest.raises(ValueError, match="actuator_count"):
            ActuatorArray(actuator_count=2)

    def test_count_exactly_3_ok(self) -> None:
        a = ActuatorArray(actuator_count=3)
        assert a.actuator_count == 3

    def test_angles_evenly_spaced(self, array4: ActuatorArray) -> None:
        angles = array4.angles
        diffs = np.diff(angles)
        expected_step = 2.0 * math.pi / 4
        np.testing.assert_array_almost_equal(diffs, [expected_step] * 3)


class TestSetAngles:
    def test_set_valid_angles(self, array4: ActuatorArray) -> None:
        new_angles = np.array([0.0, 1.0, 2.0, 3.0])
        array4.set_angles(new_angles)
        np.testing.assert_array_almost_equal(array4.angles, new_angles)

    def test_set_wrong_length_raises(self, array4: ActuatorArray) -> None:
        with pytest.raises(ValueError):
            array4.set_angles(np.array([0.0, 1.0, 2.0]))


# ---------------------------------------------------------------------------
# Grasp matrix
# ---------------------------------------------------------------------------


class TestGraspMatrix:
    def test_shape(self, array4: ActuatorArray, square: SquareBody) -> None:
        G = array4.grasp_matrix(square)
        assert G.shape == (3, 8)

    def test_shape_circle(
        self, array4: ActuatorArray, circle: CircleBody
    ) -> None:
        G = array4.grasp_matrix(circle)
        assert G.shape == (3, 8)

    def test_shape_6_actuators(self, square: SquareBody) -> None:
        a = ActuatorArray(actuator_count=6)
        G = a.grasp_matrix(square)
        assert G.shape == (3, 12)

    def test_rank_sufficient(
        self, array4: ActuatorArray, circle: CircleBody
    ) -> None:
        G = array4.grasp_matrix(circle)
        # With 4 actuators (8 cols), rank should be 3
        assert np.linalg.matrix_rank(G) == 3


# ---------------------------------------------------------------------------
# Force allocation
# ---------------------------------------------------------------------------


class TestForceAllocation:
    def test_achieve_desired_wrench_circle(
        self, array4: ActuatorArray, circle: CircleBody
    ) -> None:
        w_desired = np.array([2.0, -1.0, 0.5])
        f = array4.allocate_forces(w_desired, circle)
        G = array4.grasp_matrix(circle)
        w_actual = G @ f
        np.testing.assert_array_almost_equal(w_actual, w_desired, decimal=10)

    def test_achieve_desired_wrench_square(
        self, array4: ActuatorArray, square: SquareBody
    ) -> None:
        w_desired = np.array([1.0, 0.0, 0.0])
        f = array4.allocate_forces(w_desired, square)
        G = array4.grasp_matrix(square)
        w_actual = G @ f
        np.testing.assert_array_almost_equal(w_actual, w_desired, decimal=10)

    def test_zero_wrench_gives_zero_forces(
        self, array4: ActuatorArray, circle: CircleBody
    ) -> None:
        f = array4.allocate_forces(np.zeros(3), circle)
        np.testing.assert_array_almost_equal(f, np.zeros(8), decimal=10)

    def test_output_shape(
        self, array4: ActuatorArray, circle: CircleBody
    ) -> None:
        f = array4.allocate_forces(np.array([1.0, 0.0, 0.0]), circle)
        assert f.shape == (8,)


# ---------------------------------------------------------------------------
# Actuator positions
# ---------------------------------------------------------------------------


class TestActuatorPositions:
    def test_shape(self, array4: ActuatorArray, circle: CircleBody) -> None:
        positions = array4.actuator_positions(circle)
        assert positions.shape == (4, 2)

    def test_distance_from_body(
        self, array4: ActuatorArray, circle: CircleBody
    ) -> None:
        positions = array4.actuator_positions(circle)
        expected_dist = circle.bounding_radius + 0.05
        for pos in positions:
            dist = math.hypot(pos[0], pos[1])
            assert dist == pytest.approx(expected_dist, rel=1e-6)

    def test_positions_shift_with_body_translation(
        self, array4: ActuatorArray, circle: CircleBody
    ) -> None:
        p0 = array4.actuator_positions(circle)
        circle.reset(x=1.0, y=2.0)
        p1 = array4.actuator_positions(circle)
        diff = p1 - p0
        np.testing.assert_array_almost_equal(diff, np.full((4, 2), [1.0, 2.0]))


# ---------------------------------------------------------------------------
# Apply to body
# ---------------------------------------------------------------------------


class TestApplyToBody:
    def test_apply_moves_body(
        self, array4: ActuatorArray, circle: CircleBody
    ) -> None:
        w_desired = np.array([10.0, 0.0, 0.0])
        forces = array4.allocate_forces(w_desired, circle)
        array4.apply_to_body(forces, circle)
        circle.step(dt=1.0)
        # Should have moved in x direction
        assert circle.pose[0] > 0.0

    def test_apply_zero_forces_no_motion(
        self, array4: ActuatorArray, circle: CircleBody
    ) -> None:
        array4.apply_to_body(np.zeros(8), circle)
        circle.step(dt=1.0)
        np.testing.assert_array_almost_equal(circle.pose, [0.0, 0.0, 0.0])
