"""Tests for RRPRaceRobot per-DOF velocity limits."""

from __future__ import annotations

import numpy as np
import pytest

pygame = pytest.importorskip("pygame")  # skip whole module if no display libs

from arco.tools.simulator.main.rrp import RRPRaceRobot  # noqa: E402


def test_angular_velocity_capped() -> None:
    """Angular joints q1/q2 must respect max_vel_ang."""
    r = RRPRaceRobot(np.zeros(3), max_vel_ang=1.0, max_vel_lin=0.5)
    # large angular error: velocity should be capped at 1.0 rad/s
    for _ in range(50):
        r.step(np.array([100.0, 100.0, 0.0]), dt=0.05)
    assert abs(r.vel[0]) <= 1.0 + 1e-9
    assert abs(r.vel[1]) <= 1.0 + 1e-9


def test_linear_velocity_capped() -> None:
    """Prismatic joint z must respect max_vel_lin."""
    r = RRPRaceRobot(np.zeros(3), max_vel_ang=1.0, max_vel_lin=0.5)
    # large z error: velocity should be capped at 0.5 m/s
    for _ in range(50):
        r.step(np.array([0.0, 0.0, 100.0]), dt=0.05)
    assert abs(r.vel[2]) <= 0.5 + 1e-9


def test_angular_and_linear_independent() -> None:
    """Angular and linear limits must be enforced independently."""
    r = RRPRaceRobot(np.zeros(3), max_vel_ang=2.0, max_vel_lin=1.0)
    for _ in range(100):
        r.step(np.array([5.0, 5.0, 5.0]), dt=0.05)
    assert abs(r.vel[0]) <= 2.0 + 1e-9
    assert abs(r.vel[1]) <= 2.0 + 1e-9
    assert abs(r.vel[2]) <= 1.0 + 1e-9
