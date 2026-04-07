"""Tests for arco.kinematics.rr — RRRobot two-link planar arm."""

from __future__ import annotations

import math

import pytest

from arco.kinematics import RRRobot

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def robot() -> RRRobot:
    """Default robot with l1=1.0, l2=0.8."""
    return RRRobot(l1=1.0, l2=0.8)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestRRRobotInit:
    def test_default_link_lengths(self) -> None:
        r = RRRobot()
        assert r.l1 == pytest.approx(1.0)
        assert r.l2 == pytest.approx(0.8)

    def test_custom_link_lengths(self) -> None:
        r = RRRobot(l1=2.0, l2=1.5)
        assert r.l1 == pytest.approx(2.0)
        assert r.l2 == pytest.approx(1.5)

    def test_zero_l1_raises(self) -> None:
        with pytest.raises(ValueError, match="l1"):
            RRRobot(l1=0.0, l2=0.5)

    def test_negative_l1_raises(self) -> None:
        with pytest.raises(ValueError, match="l1"):
            RRRobot(l1=-1.0, l2=0.5)

    def test_zero_l2_raises(self) -> None:
        with pytest.raises(ValueError, match="l2"):
            RRRobot(l1=1.0, l2=0.0)

    def test_negative_l2_raises(self) -> None:
        with pytest.raises(ValueError, match="l2"):
            RRRobot(l1=1.0, l2=-0.5)


# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------


class TestForwardKinematics:
    def test_both_joints_zero(self, robot: RRRobot) -> None:
        x, y = robot.forward_kinematics(0.0, 0.0)
        assert x == pytest.approx(1.8)
        assert y == pytest.approx(0.0)

    def test_q1_pi_over_2_q2_zero(self, robot: RRRobot) -> None:
        x, y = robot.forward_kinematics(math.pi / 2, 0.0)
        assert x == pytest.approx(0.0, abs=1e-9)
        assert y == pytest.approx(1.8)

    def test_q1_pi_over_4_q2_pi_over_4(self, robot: RRRobot) -> None:
        q1 = math.pi / 4
        q2 = math.pi / 4
        # joint2 at (l1*cos(pi/4), l1*sin(pi/4)) = (√2/2, √2/2)
        # ee at joint2 + l2*(cos(pi/2), sin(pi/2)) = joint2 + (0, 0.8)
        j2x = 1.0 * math.cos(q1)
        j2y = 1.0 * math.sin(q1)
        expected_x = j2x + 0.8 * math.cos(q1 + q2)
        expected_y = j2y + 0.8 * math.sin(q1 + q2)
        x, y = robot.forward_kinematics(q1, q2)
        assert x == pytest.approx(expected_x, rel=1e-9)
        assert y == pytest.approx(expected_y, rel=1e-9)

    def test_q1_pi_q2_zero(self, robot: RRRobot) -> None:
        x, y = robot.forward_kinematics(math.pi, 0.0)
        assert x == pytest.approx(-1.8)
        assert y == pytest.approx(0.0, abs=1e-9)

    def test_returns_tuple(self, robot: RRRobot) -> None:
        result = robot.forward_kinematics(0.0, 0.0)
        assert isinstance(result, tuple)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Inverse kinematics
# ---------------------------------------------------------------------------


class TestInverseKinematics:
    def test_roundtrip_recovers_position(self, robot: RRRobot) -> None:
        for tx, ty in [(1.2, 0.3), (0.5, 1.1), (-0.8, 0.9), (1.5, 0.0)]:
            sols = robot.inverse_kinematics(tx, ty)
            assert len(sols) >= 1, f"No solution for ({tx}, {ty})"
            for q1, q2 in sols:
                rx, ry = robot.forward_kinematics(q1, q2)
                assert rx == pytest.approx(tx, abs=1e-8)
                assert ry == pytest.approx(ty, abs=1e-8)

    def test_outside_workspace_returns_empty(self, robot: RRRobot) -> None:
        r_max = robot.workspace_radius()
        sols = robot.inverse_kinematics(r_max + 1.0, 0.0)
        assert sols == []

    def test_inside_inner_radius_returns_empty(self) -> None:
        r = RRRobot(l1=1.0, l2=0.5)
        r_min = abs(r.l1 - r.l2)
        # A point strictly inside the inner hole
        if r_min > 1e-3:
            sols = r.inverse_kinematics(r_min * 0.1, 0.0)
            assert sols == []

    def test_max_reach_gives_solution(self, robot: RRRobot) -> None:
        r_max = robot.workspace_radius()
        sols = robot.inverse_kinematics(r_max, 0.0)
        assert len(sols) >= 1

    def test_two_solutions_exist_interior(self, robot: RRRobot) -> None:
        sols = robot.inverse_kinematics(1.0, 0.5)
        assert len(sols) == 2

    def test_solutions_are_tuples_of_floats(self, robot: RRRobot) -> None:
        sols = robot.inverse_kinematics(1.0, 0.0)
        for sol in sols:
            assert isinstance(sol, tuple)
            assert len(sol) == 2
            assert all(isinstance(v, float) for v in sol)

    def test_near_boundary_roundtrip(self, robot: RRRobot) -> None:
        r_max = robot.workspace_radius()
        target = r_max * 0.99
        sols = robot.inverse_kinematics(target, 0.0)
        assert len(sols) >= 1
        for q1, q2 in sols:
            rx, ry = robot.forward_kinematics(q1, q2)
            assert math.hypot(rx - target, ry) < 1e-6

    def test_negative_coordinates_roundtrip(self, robot: RRRobot) -> None:
        tx, ty = -0.8, -0.7
        sols = robot.inverse_kinematics(tx, ty)
        assert len(sols) >= 1
        for q1, q2 in sols:
            rx, ry = robot.forward_kinematics(q1, q2)
            assert rx == pytest.approx(tx, abs=1e-8)
            assert ry == pytest.approx(ty, abs=1e-8)


# ---------------------------------------------------------------------------
# Link segments
# ---------------------------------------------------------------------------


class TestLinkSegments:
    def test_zero_config_segments(self, robot: RRRobot) -> None:
        origin, j2, ee = robot.link_segments(0.0, 0.0)
        assert origin == (0.0, 0.0)
        assert j2 == pytest.approx((1.0, 0.0))
        assert ee == pytest.approx((1.8, 0.0))

    def test_q1_pi_over_2_segments(self, robot: RRRobot) -> None:
        origin, j2, ee = robot.link_segments(math.pi / 2, 0.0)
        assert origin == (0.0, 0.0)
        assert j2[0] == pytest.approx(0.0, abs=1e-9)
        assert j2[1] == pytest.approx(1.0)
        assert ee[0] == pytest.approx(0.0, abs=1e-9)
        assert ee[1] == pytest.approx(1.8)

    def test_returns_three_tuples(self, robot: RRRobot) -> None:
        result = robot.link_segments(0.1, -0.2)
        assert len(result) == 3
        for pt in result:
            assert isinstance(pt, tuple)
            assert len(pt) == 2

    def test_ee_matches_forward_kinematics(self, robot: RRRobot) -> None:
        q1, q2 = 0.7, -1.1
        _, _, ee = robot.link_segments(q1, q2)
        fk = robot.forward_kinematics(q1, q2)
        assert ee == pytest.approx(fk, rel=1e-12)


# ---------------------------------------------------------------------------
# Workspace helpers
# ---------------------------------------------------------------------------


class TestWorkspaceHelpers:
    def test_workspace_radius(self, robot: RRRobot) -> None:
        assert robot.workspace_radius() == pytest.approx(1.8)

    def test_workspace_annulus(self, robot: RRRobot) -> None:
        r_min, r_max = robot.workspace_annulus()
        assert r_min == pytest.approx(0.2)
        assert r_max == pytest.approx(1.8)

    def test_workspace_annulus_equal_links(self) -> None:
        r = RRRobot(l1=1.0, l2=1.0)
        r_min, r_max = r.workspace_annulus()
        assert r_min == pytest.approx(0.0, abs=1e-12)
        assert r_max == pytest.approx(2.0)

    def test_workspace_radius_matches_annulus_max(
        self, robot: RRRobot
    ) -> None:
        _, r_max = robot.workspace_annulus()
        assert robot.workspace_radius() == pytest.approx(r_max)
