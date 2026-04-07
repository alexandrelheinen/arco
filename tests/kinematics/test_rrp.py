"""Tests for arco.kinematics.rrp — RRPRobot SCARA-like arm."""

from __future__ import annotations

import math

import pytest

from arco.kinematics import RRPRobot

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def robot() -> RRPRobot:
    """Default robot: l1=1.0, l2=0.8, z_min=0.0, z_max=4.0."""
    return RRPRobot(l1=1.0, l2=0.8, z_min=0.0, z_max=4.0)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestRRPRobotInit:
    def test_default_values(self) -> None:
        r = RRPRobot()
        assert r.l1 == pytest.approx(1.0)
        assert r.l2 == pytest.approx(0.8)
        assert r.z_min == pytest.approx(0.0)
        assert r.z_max == pytest.approx(4.0)

    def test_custom_values(self) -> None:
        r = RRPRobot(l1=1.5, l2=1.2, z_min=0.5, z_max=6.0)
        assert r.l1 == pytest.approx(1.5)
        assert r.l2 == pytest.approx(1.2)
        assert r.z_min == pytest.approx(0.5)
        assert r.z_max == pytest.approx(6.0)

    def test_zero_l1_raises(self) -> None:
        with pytest.raises(ValueError, match="l1"):
            RRPRobot(l1=0.0, l2=0.8)

    def test_negative_l1_raises(self) -> None:
        with pytest.raises(ValueError, match="l1"):
            RRPRobot(l1=-1.0, l2=0.8)

    def test_zero_l2_raises(self) -> None:
        with pytest.raises(ValueError, match="l2"):
            RRPRobot(l1=1.0, l2=0.0)

    def test_negative_l2_raises(self) -> None:
        with pytest.raises(ValueError, match="l2"):
            RRPRobot(l1=1.0, l2=-0.5)

    def test_z_max_equal_z_min_raises(self) -> None:
        with pytest.raises(ValueError, match="z_max"):
            RRPRobot(z_min=2.0, z_max=2.0)

    def test_z_max_less_than_z_min_raises(self) -> None:
        with pytest.raises(ValueError, match="z_max"):
            RRPRobot(z_min=3.0, z_max=1.0)


# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------


class TestForwardKinematics:
    def test_all_joints_zero(self, robot: RRPRobot) -> None:
        x, y, z = robot.forward_kinematics(0.0, 0.0, 0.0)
        assert x == pytest.approx(1.8)
        assert y == pytest.approx(0.0)
        assert z == pytest.approx(0.0)

    def test_z_passes_through_unchanged(self, robot: RRPRobot) -> None:
        _, _, z_out = robot.forward_kinematics(0.5, -0.3, 2.7)
        assert z_out == pytest.approx(2.7)

    def test_q1_pi_over_2_gives_correct_xy(self, robot: RRPRobot) -> None:
        x, y, z = robot.forward_kinematics(math.pi / 2, 0.0, 1.0)
        assert x == pytest.approx(0.0, abs=1e-9)
        assert y == pytest.approx(1.8)
        assert z == pytest.approx(1.0)

    def test_q1_pi_q2_zero(self, robot: RRPRobot) -> None:
        x, y, _ = robot.forward_kinematics(math.pi, 0.0, 0.0)
        assert x == pytest.approx(-1.8, abs=1e-9)
        assert y == pytest.approx(0.0, abs=1e-9)

    def test_returns_three_floats(self, robot: RRPRobot) -> None:
        result = robot.forward_kinematics(0.3, -0.5, 1.5)
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)

    def test_xy_consistent_with_rr_kinematics(self, robot: RRPRobot) -> None:
        """XY portion must match the pure RR formula."""
        from arco.kinematics import RRRobot

        rr = RRRobot(l1=robot.l1, l2=robot.l2)
        for q1, q2, z in [
            (0.4, 0.6, 0.5),
            (-1.0, 1.2, 3.0),
            (math.pi / 3, -math.pi / 4, 2.0),
        ]:
            xp, yp, zp = robot.forward_kinematics(q1, q2, z)
            xr, yr = rr.forward_kinematics(q1, q2)
            assert xp == pytest.approx(xr, rel=1e-12)
            assert yp == pytest.approx(yr, rel=1e-12)
            assert zp == pytest.approx(z)


# ---------------------------------------------------------------------------
# Inverse kinematics XY
# ---------------------------------------------------------------------------


class TestInverseKinematicsXY:
    def test_roundtrip_recovers_xy(self, robot: RRPRobot) -> None:
        for tx, ty in [(1.2, 0.3), (0.5, 1.1), (-0.8, 0.9), (1.5, 0.0)]:
            sols = robot.inverse_kinematics_xy(tx, ty)
            assert len(sols) >= 1, f"No solution for ({tx}, {ty})"
            for q1, q2 in sols:
                x, y, _ = robot.forward_kinematics(q1, q2, 0.0)
                assert x == pytest.approx(tx, abs=1e-8)
                assert y == pytest.approx(ty, abs=1e-8)

    def test_outside_workspace_empty(self, robot: RRPRobot) -> None:
        r_max = robot.workspace_radius()
        assert robot.inverse_kinematics_xy(r_max + 1.0, 0.0) == []

    def test_two_solutions_interior(self, robot: RRPRobot) -> None:
        sols = robot.inverse_kinematics_xy(1.0, 0.5)
        assert len(sols) == 2

    def test_solutions_are_float_tuples(self, robot: RRPRobot) -> None:
        for sol in robot.inverse_kinematics_xy(1.0, 0.0):
            assert isinstance(sol, tuple)
            assert len(sol) == 2
            assert all(isinstance(v, float) for v in sol)


# ---------------------------------------------------------------------------
# Link segments
# ---------------------------------------------------------------------------


class TestLinkSegments:
    def test_zero_config_segments(self, robot: RRPRobot) -> None:
        origin, j2, ee = robot.link_segments(0.0, 0.0, 0.0)
        assert origin == (0.0, 0.0, 0.0)
        assert j2 == pytest.approx((1.0, 0.0, 0.0))
        assert ee == pytest.approx((1.8, 0.0, 0.0))

    def test_z_propagates_to_all_points(self, robot: RRPRobot) -> None:
        origin, j2, ee = robot.link_segments(0.0, 0.0, 2.5)
        assert origin[2] == pytest.approx(2.5)
        assert j2[2] == pytest.approx(2.5)
        assert ee[2] == pytest.approx(2.5)

    def test_returns_three_3d_tuples(self, robot: RRPRobot) -> None:
        result = robot.link_segments(0.1, -0.2, 1.0)
        assert len(result) == 3
        for pt in result:
            assert isinstance(pt, tuple)
            assert len(pt) == 3

    def test_ee_matches_forward_kinematics(self, robot: RRPRobot) -> None:
        q1, q2, z = 0.7, -1.1, 1.8
        _, _, ee = robot.link_segments(q1, q2, z)
        fk = robot.forward_kinematics(q1, q2, z)
        assert ee == pytest.approx(fk, rel=1e-12)

    def test_q1_pi_over_2_segments(self, robot: RRPRobot) -> None:
        origin, j2, ee = robot.link_segments(math.pi / 2, 0.0, 1.0)
        assert origin == (0.0, 0.0, 1.0)
        assert j2[0] == pytest.approx(0.0, abs=1e-9)
        assert j2[1] == pytest.approx(1.0)
        assert j2[2] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Workspace helpers
# ---------------------------------------------------------------------------


class TestWorkspaceHelpers:
    def test_workspace_radius(self, robot: RRPRobot) -> None:
        assert robot.workspace_radius() == pytest.approx(1.8)

    def test_workspace_annulus_equal_links(self) -> None:
        r = RRPRobot(l1=1.0, l2=1.0, z_max=4.0)
        r_min, r_max = r.workspace_annulus()
        assert r_min == pytest.approx(0.0, abs=1e-9)
        assert r_max == pytest.approx(2.0)

    def test_workspace_annulus_unequal_links(self, robot: RRPRobot) -> None:
        r_min, r_max = robot.workspace_annulus()
        assert r_min == pytest.approx(0.2)
        assert r_max == pytest.approx(1.8)
