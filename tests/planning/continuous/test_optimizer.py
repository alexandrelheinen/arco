"""Tests for TrajectoryOptimizer."""

import math
import warnings

import numpy as np
import pytest

from arco.guidance.vehicle import DubinsVehicle
from arco.mapping import KDTreeOccupancy
from arco.planning.continuous import TrajectoryOptimizer, TrajectoryResult

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

BOUNDS_2D = [(0.0, 20.0), (0.0, 20.0)]


def _free_occupancy(clearance=0.3):
    """Effectively free-space occupancy (single far obstacle)."""
    return KDTreeOccupancy([[200.0, 200.0]], clearance=clearance)


def _obstacle_occupancy(clearance=1.0):
    """Occupancy with an obstacle cluster near the straight-line path."""
    pts = [[5.0, y] for y in np.arange(4.0, 8.0, 0.5)]
    return KDTreeOccupancy(pts, clearance=clearance)


def _straight_path(n_segments=4):
    """Reference path: straight horizontal line divided into n_segments."""
    return [np.array([i * 4.0, 5.0]) for i in range(n_segments + 1)]


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_optimizer_construction_invalid_speed():
    occ = _free_occupancy()
    with pytest.raises(ValueError, match="cruise_speed must be positive"):
        TrajectoryOptimizer(occ, cruise_speed=0.0)


def test_optimizer_construction_negative_speed():
    occ = _free_occupancy()
    with pytest.raises(ValueError, match="cruise_speed must be positive"):
        TrajectoryOptimizer(occ, cruise_speed=-1.0)


def test_optimizer_optimize_too_short_path():
    occ = _free_occupancy()
    opt = TrajectoryOptimizer(occ)
    with pytest.raises(ValueError, match="at least two waypoints"):
        opt.optimize([np.array([0.0, 0.0])])


# ---------------------------------------------------------------------------
# Cost function evaluation
# ---------------------------------------------------------------------------














# ---------------------------------------------------------------------------
# Dynamics penalty in cost function
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# Stage 1 → Stage 2 pipeline
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# Full optimization scenario
# ---------------------------------------------------------------------------


def test_optimize_returns_correct_structure():
    """optimize() must return N+1 states, N commands, N durations."""
    occ = _free_occupancy()
    n = 4
    ref = _straight_path(n_segments=n)
    opt = TrajectoryOptimizer(occ, cruise_speed=2.0)
    result = opt.optimize(ref)

    assert isinstance(result, TrajectoryResult)
    assert len(result.states) == n + 1
    assert len(result.commands) == n
    assert len(result.durations) == n
    assert all(d > 0 for d in result.durations)
    assert math.isfinite(result.cost)


def test_optimize_endpoints_fixed():
    """Start and end waypoints must match the reference path endpoints."""
    occ = _free_occupancy()
    ref = _straight_path(n_segments=3)
    opt = TrajectoryOptimizer(occ, cruise_speed=2.0)
    result = opt.optimize(ref)

    np.testing.assert_allclose(result.states[0], ref[0])
    np.testing.assert_allclose(result.states[-1], ref[-1])


def test_optimize_free_space_total_time_near_optimal():
    """In free space, total time should be close to path_length / speed."""
    occ = _free_occupancy()
    ref = _straight_path(n_segments=4)
    cruise = 2.0
    opt = TrajectoryOptimizer(
        occ,
        cruise_speed=cruise,
        weight_time=10.0,
        weight_deviation=0.01,
        weight_velocity=1.0,
        weight_collision=0.0,
    )
    result = opt.optimize(ref)

    path_length = sum(
        float(np.linalg.norm(ref[i + 1] - ref[i])) for i in range(len(ref) - 1)
    )
    expected_time = path_length / cruise
    total_time = sum(result.durations)

    # Allow ±50% tolerance — optimizer balances time vs other terms
    assert total_time < expected_time * 2.0
    assert total_time > 0.0


def test_optimize_with_inverse_kinematics():
    """optimize() with IK callable should return commands from the IK."""
    vehicle = DubinsVehicle(max_speed=5.0, max_turn_rate=2.0)
    occ = _free_occupancy()
    ref = _straight_path(n_segments=2)
    opt = TrajectoryOptimizer(occ, cruise_speed=2.0)

    result = opt.optimize(ref, inverse_kinematics=vehicle.inverse_kinematics)

    for cmd in result.commands:
        assert cmd.shape == (2,)
        assert np.all(np.isfinite(cmd))


def test_optimize_single_segment():
    """Optimizer must handle a 2-waypoint (single-segment) reference path."""
    occ = _free_occupancy()
    ref = [np.array([0.0, 0.0]), np.array([6.0, 0.0])]
    opt = TrajectoryOptimizer(occ, cruise_speed=2.0)
    result = opt.optimize(ref)

    assert len(result.states) == 2
    assert len(result.durations) == 1
    assert result.durations[0] > 0


# ---------------------------------------------------------------------------
# TrajectoryResult.is_feasible — default and flag semantics
# ---------------------------------------------------------------------------


def test_result_is_feasible_default_true():
    """TrajectoryResult.is_feasible must default to True."""
    result = TrajectoryResult()
    assert result.is_feasible is True


def test_optimize_is_feasible_true_when_no_constraints():
    """Without speed limits or feasibility callable, result must be feasible."""
    occ = _free_occupancy()
    ref = _straight_path(n_segments=2)
    opt = TrajectoryOptimizer(occ, cruise_speed=2.0)
    result = opt.optimize(ref)
    assert result.is_feasible is True


def test_optimize_is_feasible_false_when_max_speed_violated():
    """result.is_feasible must be False when any segment exceeds max_speed."""
    occ = _free_occupancy()
    ref = [np.array([0.0, 0.0]), np.array([100.0, 0.0])]  # long path
    # Set max_speed so tight that the optimizer cannot respect it: 0.001 m/s
    opt = TrajectoryOptimizer(
        occ,
        cruise_speed=2.0,
        max_speed=0.001,
        weight_dynamics=0.0,  # disable penalty so violation remains
        max_iter=1,  # force early stop with initial guess
    )
    result = opt.optimize(ref)
    assert result.is_feasible is False


def test_optimize_is_feasible_false_when_feasibility_callable_rejects():
    """result.is_feasible must be False when feasibility callable rejects."""
    occ = _free_occupancy()
    ref = _straight_path(n_segments=2)
    opt = TrajectoryOptimizer(occ, cruise_speed=2.0)

    result = opt.optimize(ref, feasibility=lambda _state: False)
    assert result.is_feasible is False


def test_optimize_feasibility_callable_receives_derived_state():
    """feasibility callable must receive a 5-element derived state (x,y,θ,v,ω)."""
    occ = _free_occupancy()
    ref = _straight_path(n_segments=2)
    opt = TrajectoryOptimizer(occ, cruise_speed=2.0)

    received: list[np.ndarray] = []

    def capture_and_accept(state: np.ndarray) -> bool:
        received.append(state.copy())
        return True

    opt.optimize(ref, feasibility=capture_and_accept)
    assert len(received) > 0
    for st in received:
        assert st.shape == (5,)  # (x, y, θ, v, ω)


# ---------------------------------------------------------------------------
# Unit tests — _compute_derived_states helper
# ---------------------------------------------------------------------------












# ---------------------------------------------------------------------------
# Unit tests — _check_speed_bounds helper
# ---------------------------------------------------------------------------












# ---------------------------------------------------------------------------
# DubinsPrimitive turning-radius constraint via optimizer
# ---------------------------------------------------------------------------


def test_optimize_dubins_primitive_turning_radius_constraint():
    """Optimizer must mark infeasible when the DubinsPrimitive constraint fails.

    A path with a very sharp turn produces a large turn rate at the interior
    waypoint.  When the speed is high enough that |v/ω| < turning_radius, the
    DubinsPrimitive.is_feasible check should flag the result as infeasible.
    """
    from arco.guidance.primitive.dubins import DubinsPrimitive

    # Very large turning radius → tight turns are infeasible
    prim = DubinsPrimitive(turning_radius=100.0)
    occ = _free_occupancy()
    # Path makes a nearly 90-degree turn at the interior waypoint
    ref = [
        np.array([0.0, 0.0]),
        np.array([5.0, 0.0]),
        np.array([5.0, 5.0]),
    ]
    opt = TrajectoryOptimizer(
        occ,
        cruise_speed=5.0,
        weight_dynamics=0.0,  # no penalty; just flag check
        max_iter=1,  # force early stop; we want the check, not convergence
    )
    result = opt.optimize(ref, feasibility=prim.is_feasible)
    assert result.is_feasible is False


def test_feasibility_feasible_state():
    """is_feasible must return True for a valid 5-element state."""
    vehicle = DubinsVehicle(max_speed=5.0, min_speed=0.0, max_turn_rate=1.0)
    state_ok = np.array([1.0, 2.0, 0.5, 3.0, 0.5])
    assert vehicle.is_feasible(state_ok) is True


def test_feasibility_overspeed_rejected():
    """is_feasible must return False when speed exceeds max_speed."""
    vehicle = DubinsVehicle(max_speed=5.0, max_turn_rate=1.0)
    state_fast = np.array([0.0, 0.0, 0.0, 10.0, 0.0])
    assert vehicle.is_feasible(state_fast) is False


def test_feasibility_underspeed_rejected():
    """is_feasible must return False when speed is below min_speed."""
    vehicle = DubinsVehicle(max_speed=5.0, min_speed=1.0, max_turn_rate=1.0)
    state_slow = np.array([0.0, 0.0, 0.0, 0.5, 0.0])
    assert vehicle.is_feasible(state_slow) is False


def test_feasibility_overturn_rejected():
    """is_feasible must return False when |turn_rate| exceeds max."""
    vehicle = DubinsVehicle(max_speed=5.0, max_turn_rate=1.0)
    state_spin = np.array([0.0, 0.0, 0.0, 2.0, 5.0])
    assert vehicle.is_feasible(state_spin) is False


def test_feasibility_kinematic_state_always_true():
    """A 3-element kinematic state is always feasible."""
    vehicle = DubinsVehicle(max_speed=5.0, max_turn_rate=1.0)
    state_3d = np.array([1.0, 2.0, 0.5])
    assert vehicle.is_feasible(state_3d) is True


# ---------------------------------------------------------------------------
# DubinsVehicle.is_feasible via optimizer — derived state integration
# ---------------------------------------------------------------------------


def test_optimize_with_vehicle_feasibility_sets_flag():
    """optimizer with vehicle.is_feasible must set is_feasible correctly."""
    # Vehicle with a very low max_speed so the optimizer cannot satisfy it
    vehicle = DubinsVehicle(
        max_speed=0.001, min_speed=0.0, max_turn_rate=100.0
    )
    occ = _free_occupancy()
    ref = _straight_path(n_segments=2)
    # weight_dynamics=0 so penalty doesn't steer optimizer; just flag check
    opt = TrajectoryOptimizer(
        occ, cruise_speed=2.0, weight_dynamics=0.0, max_iter=1
    )
    result = opt.optimize(ref, feasibility=vehicle.is_feasible)
    # Derived speed will be >> vehicle.max_speed → infeasible
    assert result.is_feasible is False


# ---------------------------------------------------------------------------
# Inverse kinematics — DubinsVehicle
# ---------------------------------------------------------------------------


def test_ik_returns_valid_commands():
    """inverse_kinematics must return a 2-element command array."""
    vehicle = DubinsVehicle(max_speed=5.0, max_turn_rate=2.0)
    cmd = vehicle.inverse_kinematics(
        np.array([0.0, 0.0]),
        np.array([4.0, 0.0]),
        speed=2.0,
        duration=2.0,
    )
    assert cmd.shape == (2,)
    assert np.all(np.isfinite(cmd))


def test_ik_speed_saturated():
    """inverse_kinematics must not return speed above max_speed."""
    vehicle = DubinsVehicle(max_speed=3.0)
    cmd = vehicle.inverse_kinematics(
        np.array([0.0, 0.0]),
        np.array([10.0, 0.0]),
        speed=100.0,
        duration=1.0,
    )
    assert cmd[0] <= vehicle.max_speed + 1e-9


def test_ik_turn_rate_saturated():
    """inverse_kinematics must not exceed max_turn_rate."""
    vehicle = DubinsVehicle(max_speed=5.0, max_turn_rate=0.5)
    cmd = vehicle.inverse_kinematics(
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 4.0]),
        speed=2.0,
        duration=0.001,  # Very short: requires huge turn rate → saturate
    )
    assert abs(cmd[1]) <= vehicle.max_turn_rate + 1e-9


def test_ik_heading_toward_goal():
    """Turn rate sign should steer toward the goal heading."""
    vehicle = DubinsVehicle(max_speed=5.0, max_turn_rate=2.0)
    # Heading east (0 rad), goal is north-east → should turn left (positive)
    start = np.array([0.0, 0.0, 0.0])
    goal = np.array([3.0, 3.0])
    cmd = vehicle.inverse_kinematics(start, goal, speed=2.0, duration=2.0)
    # Expected: target heading ≈ pi/4, current heading = 0 → turn left (>0)
    assert cmd[1] > 0
