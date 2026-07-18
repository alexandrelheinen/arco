"""Tests for JointSpaceMPC carrot tracking."""

from __future__ import annotations

import numpy as np
import pytest

casadi = pytest.importorskip("casadi")

from arco.control.mpc.joint_space import (  # noqa: E402
    JointSpaceMPC,
    JointSpaceMPCConfig,
)
from tests.control.mpc.conftest import RectOccupancy  # noqa: E402


def test_joint_space_mpc_tracks_carrot() -> None:
    cfg = JointSpaceMPCConfig(horizon_step_count=10, dt=0.05)
    mpc = JointSpaceMPC(
        max_vel=np.array([1.0, 1.0, 1.0]),
        max_acc=np.array([2.0, 2.0, 2.0]),
        config=cfg,
    )
    mpc.reset(np.array([0.0, 0.0, 0.0]))
    target = np.array([1.0, 0.0, 0.0])
    for _ in range(80):
        q = mpc.step(target, dt=0.05)
    assert np.linalg.norm(q - target) < 0.15
    assert mpc.last_solver_success


def test_joint_space_mpc_avoids_box() -> None:
    # 2-D Cartesian gantry slice: box blocks the straight line to target.
    occ = RectOccupancy(0.8, 1.2, -0.4, 0.4, clearance=0.35)
    cfg = JointSpaceMPCConfig(
        horizon_step_count=12,
        dt=0.05,
        weight_obstacle=100.0,
        weight_tracking=15.0,
    )
    mpc = JointSpaceMPC(
        max_vel=np.array([1.2, 1.2]),
        max_acc=np.array([2.5, 2.5]),
        occupancy=occ,
        config=cfg,
    )
    mpc.reset(np.array([0.0, 0.0]))
    target = np.array([2.0, 0.0])
    min_clearance = float("inf")
    for _ in range(120):
        q = mpc.step(target, dt=0.05)
        dist, _ = occ.nearest_obstacle(q)
        min_clearance = min(min_clearance, dist)
    assert min_clearance >= 0.8 * occ.clearance


def test_build_joint_tracker_mpc_factory() -> None:
    from arco.simulator.sim.tracking import build_joint_tracker

    tracker = build_joint_tracker(
        max_vel=[1.0, 1.0],
        max_acc=[2.0, 2.0],
        tracker="mpc",
    )
    assert isinstance(tracker, JointSpaceMPC)
    tracker.reset(np.zeros(2))
    q = tracker.step(np.array([0.5, 0.0]), dt=0.05)
    assert q.shape == (2,)
