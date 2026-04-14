"""Tests for PPPRobot per-axis independence."""

from __future__ import annotations

import numpy as np

from arco.tools.simulator.main.ppp import PPPRobot


def test_axes_are_independent() -> None:
    """Y and Z must not be dragged along by a large X error."""
    r = PPPRobot(np.zeros(3), max_vel=3.0, max_acc=8.0)
    # Large X target, zero Y/Z target
    for _ in range(100):
        r.step(np.array([100.0, 0.0, 0.0]), dt=0.05)
    # Y and Z velocity must remain near 0 (not dragged by X norm)
    assert abs(r.vel[1]) < 0.1
    assert abs(r.vel[2]) < 0.1
    # X should be at max_vel
    assert abs(r.vel[0]) <= 3.0 + 1e-9


def test_velocity_cap_per_axis() -> None:
    """All axes must be individually capped at max_vel."""
    r = PPPRobot(np.zeros(3), max_vel=2.0, max_acc=100.0)
    for _ in range(200):
        r.step(np.array([1000.0, 1000.0, 1000.0]), dt=0.05)
    assert all(abs(r.vel) <= 2.0 + 1e-9)
