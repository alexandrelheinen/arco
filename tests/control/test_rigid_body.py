"""Tests for arco.control.rigid_body — RigidBody, SquareBody, CircleBody."""

from __future__ import annotations

import math

import numpy as np
import pytest

from arco.control.rigid_body import CircleBody, RigidBody, SquareBody

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def square() -> SquareBody:
    """Default 5 kg square with side 0.5 m."""
    return SquareBody(mass=5.0, side_length=0.5)


@pytest.fixture()
def circle() -> CircleBody:
    """Default 5 kg circle with radius 0.3 m."""
    return CircleBody(mass=5.0, radius=0.3)


# ---------------------------------------------------------------------------
# Init tests
# ---------------------------------------------------------------------------


class TestSquareBodyInit:
    def test_default_pose(self, square: SquareBody) -> None:
        np.testing.assert_array_almost_equal(square.pose, [0.0, 0.0, 0.0])

    def test_custom_pose(self) -> None:
        s = SquareBody(mass=1.0, side_length=0.4, x=1.0, y=2.0, psi=0.5)
        np.testing.assert_array_almost_equal(s.pose, [1.0, 2.0, 0.5])

    def test_zero_mass_raises(self) -> None:
        with pytest.raises(ValueError, match="mass"):
            SquareBody(mass=0.0, side_length=0.5)

    def test_negative_mass_raises(self) -> None:
        with pytest.raises(ValueError, match="mass"):
            SquareBody(mass=-1.0, side_length=0.5)

    def test_zero_side_raises(self) -> None:
        with pytest.raises(ValueError, match="side_length"):
            SquareBody(mass=1.0, side_length=0.0)

    def test_negative_side_raises(self) -> None:
        with pytest.raises(ValueError, match="side_length"):
            SquareBody(mass=1.0, side_length=-0.5)

    def test_mass_property(self, square: SquareBody) -> None:
        assert square.mass == pytest.approx(5.0)

    def test_side_length_property(self, square: SquareBody) -> None:
        assert square.side_length == pytest.approx(0.5)


class TestCircleBodyInit:
    def test_default_pose(self, circle: CircleBody) -> None:
        np.testing.assert_array_almost_equal(circle.pose, [0.0, 0.0, 0.0])

    def test_zero_mass_raises(self) -> None:
        with pytest.raises(ValueError, match="mass"):
            CircleBody(mass=0.0, radius=0.3)

    def test_zero_radius_raises(self) -> None:
        with pytest.raises(ValueError, match="radius"):
            CircleBody(mass=1.0, radius=0.0)

    def test_negative_radius_raises(self) -> None:
        with pytest.raises(ValueError, match="radius"):
            CircleBody(mass=1.0, radius=-0.1)

    def test_radius_property(self, circle: CircleBody) -> None:
        assert circle.radius == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Inertia and bounding radius
# ---------------------------------------------------------------------------


class TestRigidBodyInertia:
    def test_square_inertia(self, square: SquareBody) -> None:
        # I = mass * side² / 6 = 5.0 * 0.25 / 6
        expected = 5.0 * 0.5**2 / 6.0
        assert square.inertia == pytest.approx(expected)

    def test_circle_inertia(self, circle: CircleBody) -> None:
        # I = mass * radius² / 2 = 5.0 * 0.09 / 2
        expected = 5.0 * 0.3**2 / 2.0
        assert circle.inertia == pytest.approx(expected)

    def test_square_bounding_radius(self, square: SquareBody) -> None:
        expected = 0.5 * math.sqrt(2.0) / 2.0
        assert square.bounding_radius == pytest.approx(expected)

    def test_circle_bounding_radius(self, circle: CircleBody) -> None:
        assert circle.bounding_radius == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------


class TestRigidBodyPhysics:
    def test_zero_force_constant_velocity(self, square: SquareBody) -> None:
        square._velocity[:] = [1.0, 0.5, 0.1]
        square.step(dt=0.1)
        np.testing.assert_array_almost_equal(square.velocity, [1.0, 0.5, 0.1])

    def test_force_produces_acceleration(self, square: SquareBody) -> None:
        square.apply_wrench(fx=10.0, fy=0.0, torque=0.0)
        square.step(dt=1.0)
        # ax = F/m = 10/5 = 2 m/s²; vx = 2*1 = 2
        assert square.velocity[0] == pytest.approx(2.0)

    def test_torque_produces_angular_acceleration(
        self, square: SquareBody
    ) -> None:
        I = square.inertia
        square.apply_wrench(fx=0.0, fy=0.0, torque=I)
        square.step(dt=1.0)
        # alpha = torque/I = 1 rad/s²; omega = 1*1 = 1
        assert square.velocity[2] == pytest.approx(1.0)

    def test_wrench_cleared_after_step(self, square: SquareBody) -> None:
        square.apply_wrench(fx=5.0, fy=0.0, torque=0.0)
        square.step(dt=0.1)
        v0 = square.velocity[0]
        square.step(dt=0.1)
        assert square.velocity[0] == pytest.approx(v0)

    def test_position_integrates_from_velocity(
        self, square: SquareBody
    ) -> None:
        square.apply_wrench(fx=square.mass, fy=0.0, torque=0.0)
        square.step(dt=1.0)
        # After 1s: vx=1 m/s, x=1*1=1 m
        assert square.pose[0] == pytest.approx(1.0)

    def test_euler_integration_numerics(self) -> None:
        body = SquareBody(mass=2.0, side_length=1.0)
        body.apply_wrench(fx=4.0, fy=0.0, torque=0.0)
        body.step(dt=0.5)
        # ax=2, vx=1, x=0.5
        assert body.velocity[0] == pytest.approx(1.0)
        assert body.pose[0] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestRigidBodyReset:
    def test_reset_clears_velocity(self, square: SquareBody) -> None:
        square._velocity[:] = [1.0, 2.0, 3.0]
        square.reset()
        np.testing.assert_array_almost_equal(square.velocity, [0.0, 0.0, 0.0])

    def test_reset_sets_pose(self, square: SquareBody) -> None:
        square.reset(x=1.0, y=2.0, psi=0.5)
        np.testing.assert_array_almost_equal(square.pose, [1.0, 2.0, 0.5])

    def test_reset_default_pose(self, square: SquareBody) -> None:
        square.reset()
        np.testing.assert_array_almost_equal(square.pose, [0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# SquareBody corners
# ---------------------------------------------------------------------------


class TestSquareBodyCorners:
    def test_corners_shape(self, square: SquareBody) -> None:
        c = square.corners()
        assert c.shape == (4, 2)

    def test_corners_zero_pose(self, square: SquareBody) -> None:
        a = 0.5 / 2.0
        c = square.corners()
        expected = np.array([(-a, -a), (a, -a), (a, a), (-a, a)], dtype=float)
        np.testing.assert_array_almost_equal(c, expected)

    def test_corners_translated(self) -> None:
        s = SquareBody(mass=1.0, side_length=1.0, x=3.0, y=4.0)
        c = s.corners()
        # Center at (3, 4), half-side=0.5
        cx_vals = c[:, 0]
        cy_vals = c[:, 1]
        assert min(cx_vals) == pytest.approx(2.5)
        assert max(cx_vals) == pytest.approx(3.5)
        assert min(cy_vals) == pytest.approx(3.5)
        assert max(cy_vals) == pytest.approx(4.5)

    def test_corners_90_degree_rotation(self) -> None:
        s = SquareBody(mass=1.0, side_length=2.0, psi=math.pi / 2)
        c = s.corners()
        # After 90° rotation, x-extents become y-extents
        assert c.shape == (4, 2)
        # All corners should be at distance side*sqrt(2)/2 from origin
        dists = np.linalg.norm(c, axis=1)
        expected_dist = math.sqrt(2.0)
        np.testing.assert_array_almost_equal(dists, [expected_dist] * 4)

    def test_corners_distance_from_center(self, square: SquareBody) -> None:
        c = square.corners()
        dists = np.linalg.norm(c, axis=1)
        expected = square.bounding_radius
        np.testing.assert_array_almost_equal(dists, [expected] * 4)

    def test_is_abstract(self) -> None:
        assert not isinstance(RigidBody, type) or RigidBody.__abstractmethods__
