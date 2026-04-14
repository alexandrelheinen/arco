"""Tests for JointSpaceTracker."""

from __future__ import annotations

import numpy as np
import pytest

from arco.control import JointSpaceTracker
from arco.mapping import KDTreeOccupancy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAX_VEL_2D = np.array([1.0, 1.0])
_MAX_ACC_2D = np.array([2.0, 2.0])


def _free_occ() -> KDTreeOccupancy:
    return KDTreeOccupancy([[1000.0, 1000.0]], clearance=0.5)


def _blocked_occ(pos=(0.5, 0.0), clearance=0.3) -> KDTreeOccupancy:
    return KDTreeOccupancy([list(pos)], clearance=clearance)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_stores_params():
    t = JointSpaceTracker(max_vel=1.0, max_acc=2.0, proportional_gain=3.0)
    assert np.allclose(t._max_vel, [1.0])
    assert np.allclose(t._max_acc, [2.0])
    assert t._k_p == pytest.approx(3.0)
    assert t.repulsion_gain == pytest.approx(0.0)
    assert t._occ is None


def test_construction_vector_params():
    t = JointSpaceTracker(max_vel=_MAX_VEL_2D, max_acc=_MAX_ACC_2D)
    assert np.allclose(t._max_vel, _MAX_VEL_2D)
    assert np.allclose(t._max_acc, _MAX_ACC_2D)


def test_construction_zero_max_vel_raises():
    with pytest.raises(ValueError, match="max_vel must be strictly positive"):
        JointSpaceTracker(max_vel=0.0, max_acc=1.0)


def test_construction_negative_max_vel_raises():
    with pytest.raises(ValueError, match="max_vel must be strictly positive"):
        JointSpaceTracker(max_vel=[1.0, -0.5], max_acc=[1.0, 1.0])


def test_construction_zero_max_acc_raises():
    with pytest.raises(ValueError, match="max_acc must be strictly positive"):
        JointSpaceTracker(max_vel=1.0, max_acc=0.0)


def test_construction_with_occupancy():
    occ = _free_occ()
    t = JointSpaceTracker(
        max_vel=1.0, max_acc=2.0, occupancy=occ, repulsion_gain=0.5
    )
    assert t._occ is occ
    assert t.repulsion_gain == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


def test_reset_sets_q_and_zeros_vel():
    t = JointSpaceTracker(max_vel=_MAX_VEL_2D, max_acc=_MAX_ACC_2D)
    q0 = np.array([1.5, -0.3])
    t.reset(q0)
    assert np.allclose(t.q, q0)
    assert np.allclose(t.vel, np.zeros(2))


def test_reset_copies_array():
    t = JointSpaceTracker(max_vel=_MAX_VEL_2D, max_acc=_MAX_ACC_2D)
    q0 = np.array([1.0, 2.0])
    t.reset(q0)
    q0[:] = 99.0
    assert np.allclose(t.q, [1.0, 2.0])


# ---------------------------------------------------------------------------
# step() — basic correctness
# ---------------------------------------------------------------------------


def test_step_moves_toward_target():
    t = JointSpaceTracker(max_vel=_MAX_VEL_2D, max_acc=_MAX_ACC_2D)
    t.reset(np.zeros(2))
    target = np.array([5.0, 5.0])
    q1 = t.step(target, dt=0.1)
    # Should have moved toward target (positive direction)
    assert q1[0] > 0.0 and q1[1] > 0.0


def test_step_at_target_does_not_move():
    """When already at target with zero velocity, step stays put."""
    t = JointSpaceTracker(max_vel=_MAX_VEL_2D, max_acc=_MAX_ACC_2D)
    t.reset(np.array([3.0, 3.0]))
    q1 = t.step(np.array([3.0, 3.0]), dt=0.1)
    assert np.allclose(q1, [3.0, 3.0], atol=1e-10)


def test_step_velocity_saturated_by_max_vel():
    """With very high k_p, velocity should be clamped to max_vel."""
    t = JointSpaceTracker(
        max_vel=np.array([0.5, 0.5]),
        max_acc=np.array([100.0, 100.0]),  # high acc so it doesn't limit us
        proportional_gain=1000.0,
    )
    t.reset(np.zeros(2))
    t.step(np.array([100.0, 100.0]), dt=0.1)
    assert np.all(np.abs(t.vel) <= 0.5 + 1e-9)


def test_step_acceleration_limited():
    """Velocity change per step is bounded by max_acc * dt."""
    max_acc = np.array([1.0, 1.0])
    t = JointSpaceTracker(
        max_vel=np.array([10.0, 10.0]),
        max_acc=max_acc,
        proportional_gain=100.0,
    )
    t.reset(np.zeros(2))
    dt = 0.1
    t.step(np.array([100.0, 100.0]), dt=dt)
    # Velocity should not exceed max_acc * dt from start (vel was 0)
    assert np.all(np.abs(t.vel) <= max_acc * dt + 1e-9)


def test_step_returns_new_q():
    t = JointSpaceTracker(max_vel=_MAX_VEL_2D, max_acc=_MAX_ACC_2D)
    t.reset(np.zeros(2))
    q1 = t.step(np.array([1.0, 1.0]), dt=0.1)
    assert isinstance(q1, np.ndarray)
    assert q1.shape == (2,)


def test_step_updates_internal_q():
    t = JointSpaceTracker(max_vel=_MAX_VEL_2D, max_acc=_MAX_ACC_2D)
    t.reset(np.zeros(2))
    q1 = t.step(np.array([1.0, 0.0]), dt=0.1)
    assert np.allclose(t.q, q1)


def test_step_returns_copy_not_alias():
    """Returned array must not alias internal state."""
    t = JointSpaceTracker(max_vel=_MAX_VEL_2D, max_acc=_MAX_ACC_2D)
    t.reset(np.zeros(2))
    q1 = t.step(np.array([1.0, 0.0]), dt=0.1)
    q1[:] = 99.0
    assert not np.allclose(t.q, [99.0, 99.0])


# ---------------------------------------------------------------------------
# step() — convergence
# ---------------------------------------------------------------------------


def test_step_converges_to_target_in_free_space():
    """After many steps, tracker should converge close to the target."""
    t = JointSpaceTracker(
        max_vel=np.array([2.0, 2.0]),
        max_acc=np.array([5.0, 5.0]),
        proportional_gain=3.0,
    )
    target = np.array([3.0, -2.0])
    t.reset(np.zeros(2))
    for _ in range(500):
        t.step(target, dt=0.05)
    assert np.allclose(t.q, target, atol=0.05)


# ---------------------------------------------------------------------------
# _repulsion_velocity()
# ---------------------------------------------------------------------------


def test_repulsion_zero_when_disabled():
    t = JointSpaceTracker(max_vel=1.0, max_acc=2.0, repulsion_gain=0.0)
    t.reset(np.zeros(1))
    rv = t._repulsion_velocity(np.zeros(1))
    assert np.allclose(rv, 0.0)


def test_repulsion_zero_without_occupancy():
    t = JointSpaceTracker(max_vel=1.0, max_acc=2.0, repulsion_gain=1.0)
    t.reset(np.zeros(1))
    rv = t._repulsion_velocity(np.zeros(1))
    assert np.allclose(rv, 0.0)


def test_repulsion_zero_outside_influence_radius():
    """No repulsion when farther than 2*clearance from obstacle."""
    occ = KDTreeOccupancy([[0.0, 0.0]], clearance=0.1)
    t = JointSpaceTracker(
        max_vel=1.0, max_acc=2.0, occupancy=occ, repulsion_gain=1.0
    )
    t.reset(np.zeros(2))
    # 5.0 >> 2*0.1 = 0.2 influence radius
    rv = t._repulsion_velocity(np.array([5.0, 5.0]))
    assert np.allclose(rv, 0.0)


def test_repulsion_nonzero_inside_influence_radius():
    """Repulsion is nonzero when inside 2*clearance."""
    occ = KDTreeOccupancy([[0.0, 0.0]], clearance=0.5)
    t = JointSpaceTracker(
        max_vel=1.0, max_acc=2.0, occupancy=occ, repulsion_gain=1.0
    )
    t.reset(np.zeros(2))
    rv = t._repulsion_velocity(np.array([0.4, 0.0]))  # inside influence=1.0
    assert not np.allclose(rv, 0.0)


def test_repulsion_points_away_from_obstacle():
    """Repulsion velocity must point away from the obstacle."""
    occ = KDTreeOccupancy([[0.0, 0.0]], clearance=0.5)
    t = JointSpaceTracker(
        max_vel=1.0, max_acc=2.0, occupancy=occ, repulsion_gain=1.0
    )
    q = np.array([0.3, 0.0])  # to the right of the obstacle at origin
    rv = t._repulsion_velocity(q)
    # Repulsion should be in the positive x direction (away from obstacle)
    assert rv[0] > 0.0


def test_repulsion_integrated_keeps_distance():
    """With repulsion enabled, tracker stays clear of a nearby obstacle."""
    # Obstacle at (1.0, 0.0), clearance=0.3 → influence=0.6
    occ = KDTreeOccupancy([[1.0, 0.0]], clearance=0.3)
    t = JointSpaceTracker(
        max_vel=np.array([1.0, 1.0]),
        max_acc=np.array([5.0, 5.0]),
        proportional_gain=2.0,
        occupancy=occ,
        repulsion_gain=1.5,
    )
    # Target passes through obstacle (from 0 to 2)
    t.reset(np.array([0.0, 0.0]))
    min_dist = float("inf")
    for _ in range(200):
        t.step(np.array([2.0, 0.0]), dt=0.05)
        dist, _ = occ.nearest_obstacle(t.q)
        min_dist = min(min_dist, dist)
    # Without repulsion the path would cross through d < clearance=0.3
    # With repulsion the tracker should stay outside (or close to) clearance
    assert min_dist >= 0.1  # repulsion keeps it from getting too close


# ---------------------------------------------------------------------------
# 3-D configuration space (PPP-style)
# ---------------------------------------------------------------------------


def test_step_3d_moves_in_all_axes():
    t = JointSpaceTracker(
        max_vel=np.array([1.0, 1.0, 0.5]),
        max_acc=np.array([2.0, 2.0, 1.0]),
    )
    t.reset(np.zeros(3))
    q1 = t.step(np.array([1.0, 2.0, 0.5]), dt=0.1)
    assert all(q1[i] >= 0.0 for i in range(3))


def test_step_3d_converges():
    t = JointSpaceTracker(
        max_vel=np.array([2.0, 2.0, 1.0]),
        max_acc=np.array([5.0, 5.0, 2.0]),
        proportional_gain=3.0,
    )
    target = np.array([1.0, -1.5, 0.8])
    t.reset(np.zeros(3))
    for _ in range(500):
        t.step(target, dt=0.05)
    assert np.allclose(t.q, target, atol=0.05)
