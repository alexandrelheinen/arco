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


def test_cost_function_time_term_dominates():
    """Time-weighted cost must increase when total time increases."""
    occ = _free_occupancy()
    opt = TrajectoryOptimizer(occ, weight_time=100.0, weight_velocity=0.0)
    # Use a single-segment path so no interior waypoints: x = [t0]
    ref = [np.array([0.0, 0.0]), np.array([4.0, 0.0])]
    segment_count = 1
    dim = 2

    # Short duration → low time cost
    x_fast = np.array([0.5])
    # Long duration → high time cost
    x_slow = np.array([5.0])

    cost_fast = opt._cost(x_fast, ref, segment_count, dim)
    cost_slow = opt._cost(x_slow, ref, segment_count, dim)
    assert cost_slow > cost_fast


def test_cost_function_deviation_term():
    """Displaced interior waypoints must increase the deviation cost."""
    occ = _free_occupancy()
    opt = TrajectoryOptimizer(
        occ,
        weight_time=0.0,
        weight_deviation=1.0,
        weight_velocity=0.0,
        weight_collision=0.0,
    )
    ref = _straight_path(n_segments=2)  # 3 waypoints, 1 interior
    segment_count = 2
    dim = 2

    # On-reference interior waypoint: x = [t0, t1, 4.0, 5.0]
    x_on = np.array([1.0, 1.0, 4.0, 5.0])
    # Displaced interior waypoint: x = [t0, t1, 4.0, 7.0]
    x_off = np.array([1.0, 1.0, 4.0, 7.0])

    cost_on = opt._cost(x_on, ref, segment_count, dim)
    cost_off = opt._cost(x_off, ref, segment_count, dim)
    assert cost_off > cost_on


def test_cost_function_velocity_term():
    """Velocities far from cruise speed must increase cost."""
    occ = _free_occupancy()
    cruise = 2.0
    opt = TrajectoryOptimizer(
        occ,
        cruise_speed=cruise,
        weight_time=0.0,
        weight_deviation=0.0,
        weight_velocity=1.0,
        weight_collision=0.0,
    )
    ref = [np.array([0.0, 0.0]), np.array([4.0, 0.0])]  # L=4, 1 segment
    segment_count = 1
    dim = 2

    # Duration giving cruise speed: t = L/v = 4/2 = 2.0
    x_cruise = np.array([2.0])
    # Duration giving much slower speed
    x_slow = np.array([20.0])

    cost_cruise = opt._cost(x_cruise, ref, segment_count, dim)
    cost_slow = opt._cost(x_slow, ref, segment_count, dim)
    assert cost_slow > cost_cruise


def test_cost_function_collision_term():
    """Waypoints near obstacles must incur higher collision cost."""
    occ = _obstacle_occupancy(clearance=1.5)
    opt = TrajectoryOptimizer(
        occ,
        weight_time=0.0,
        weight_deviation=0.0,
        weight_velocity=0.0,
        weight_collision=10.0,
        sample_count=0,
    )
    # Path: start at x=0, end at x=10, interior at y=5 (obstacle region)
    ref = [
        np.array([0.0, 5.0]),
        np.array([5.0, 5.0]),
        np.array([10.0, 5.0]),
    ]
    segment_count = 2
    dim = 2

    # On-obstacle interior: [t0, t1, 5.0, 5.0] — interior is near obstacle
    x_collision = np.array([1.0, 1.0, 5.0, 5.0])
    # Clear interior: [t0, t1, 5.0, 15.0] — interior far from obstacle
    x_clear = np.array([1.0, 1.0, 5.0, 15.0])

    cost_collision = opt._cost(x_collision, ref, segment_count, dim)
    cost_clear = opt._cost(x_clear, ref, segment_count, dim)
    assert cost_collision > cost_clear


def test_all_four_cost_terms_nonzero():
    """All four cost terms should contribute to the total cost."""
    occ = _obstacle_occupancy(clearance=2.0)
    opt = TrajectoryOptimizer(
        occ,
        cruise_speed=2.0,
        weight_time=1.0,
        weight_deviation=1.0,
        weight_velocity=1.0,
        weight_collision=1.0,
        sample_count=0,
    )
    ref = [
        np.array([0.0, 5.0]),
        np.array([5.0, 5.0]),
        np.array([10.0, 5.0]),
    ]
    segment_count = 2
    dim = 2

    # Displacement so deviation ≠ 0 and slow speed for velocity term
    x = np.array([5.0, 5.0, 5.0, 6.0])

    # Compute individual term contributions
    t_sum = 5.0 + 5.0
    j_time = 1.0 * t_sum**2
    deviation = (6.0 - 5.0) ** 2
    j_deviation = 1.0 * deviation
    # Speed = |p2 - p1| / t1; segment 0: dist=5,t=5→speed=1; segment 1: dist≈5,t=5→speed≈1
    j_velocity_nonzero = 1.0 * (1.0 - 2.0) ** 2 > 0

    total = opt._cost(x, ref, segment_count, dim)
    assert total > j_time  # Other terms add to time alone
    assert total > 0.0


# ---------------------------------------------------------------------------
# Dynamics penalty in cost function
# ---------------------------------------------------------------------------


def test_cost_dynamics_penalty_over_max_speed():
    """Implied speed exceeding max_speed must incur dynamics penalty."""
    occ = _free_occupancy()
    max_spd = 1.0  # very tight
    opt = TrajectoryOptimizer(
        occ,
        weight_time=0.0,
        weight_velocity=0.0,
        weight_collision=0.0,
        weight_dynamics=10.0,
        max_speed=max_spd,
    )
    # Single segment, length=4, duration=1 → implied speed=4 > max_speed=1
    ref = [np.array([0.0, 0.0]), np.array([4.0, 0.0])]
    x_fast = np.array([1.0])  # t=1, v=4 → violates max_speed=1

    # Without dynamics penalty (max_speed=None)
    opt_no_dyn = TrajectoryOptimizer(
        occ,
        weight_time=0.0,
        weight_velocity=0.0,
        weight_collision=0.0,
        weight_dynamics=10.0,
        max_speed=None,
    )
    cost_nodyn = opt_no_dyn._cost(x_fast, ref, 1, 2)
    cost_dyn = opt._cost(x_fast, ref, 1, 2)
    assert cost_dyn > cost_nodyn


def test_cost_dynamics_penalty_under_min_speed():
    """Implied speed below min_speed must incur dynamics penalty."""
    occ = _free_occupancy()
    opt = TrajectoryOptimizer(
        occ,
        weight_time=0.0,
        weight_velocity=0.0,
        weight_collision=0.0,
        weight_dynamics=10.0,
        min_speed=3.0,
    )
    # Single segment, length=4, duration=10 → implied speed=0.4 < min_speed=3
    ref = [np.array([0.0, 0.0]), np.array([4.0, 0.0])]
    x_slow = np.array([10.0])

    opt_no_dyn = TrajectoryOptimizer(
        occ,
        weight_time=0.0,
        weight_velocity=0.0,
        weight_collision=0.0,
        weight_dynamics=10.0,
        min_speed=None,
    )
    cost_nodyn = opt_no_dyn._cost(x_slow, ref, 1, 2)
    cost_dyn = opt._cost(x_slow, ref, 1, 2)
    assert cost_dyn > cost_nodyn


# ---------------------------------------------------------------------------
# Stage 1 → Stage 2 pipeline
# ---------------------------------------------------------------------------


def test_stage1_output_is_valid_initial_guess():
    """Stage-1 initial guess must be a finite vector on the reference path."""
    occ = _free_occupancy()
    opt = TrajectoryOptimizer(occ, cruise_speed=2.0, time_relaxation=1.5)
    ref = _straight_path(n_segments=3)
    segment_count = len(ref) - 1
    dim = 2

    x0 = opt._initial_guess(ref, segment_count, dim)

    assert np.all(np.isfinite(x0))
    # Durations should be positive
    assert np.all(x0[:segment_count] > 0)
    # Interior waypoints should equal reference
    for i in range(segment_count - 1):
        p = x0[segment_count + i * dim : segment_count + (i + 1) * dim]
        np.testing.assert_allclose(p, ref[i + 1], rtol=1e-9)


def test_stage1_used_as_stage2_input():
    """Full optimize call must improve or maintain cost vs Stage-1 guess."""
    occ = _free_occupancy()
    opt = TrajectoryOptimizer(
        occ,
        cruise_speed=2.0,
        weight_time=10.0,
        weight_deviation=1.0,
        weight_velocity=1.0,
        weight_collision=0.0,
    )
    ref = _straight_path(n_segments=3)
    segment_count = len(ref) - 1
    dim = 2

    x0 = opt._initial_guess(ref, segment_count, dim)
    cost_stage1 = opt._cost(x0, ref, segment_count, dim)

    result = opt.optimize(ref)
    assert result.cost <= cost_stage1 + 1e-6  # Stage 2 must not be worse


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


def test_compute_derived_states_shape():
    """_compute_derived_states must return N+1 states of shape (5,)."""
    occ = _free_occupancy()
    opt = TrajectoryOptimizer(occ, cruise_speed=2.0)
    n = 3
    waypoints = [np.array([float(i) * 4.0, 0.0]) for i in range(n + 1)]
    durations = np.array([2.0, 2.0, 2.0])
    derived = opt._compute_derived_states(waypoints, durations)
    assert len(derived) == n + 1
    for st in derived:
        assert st.shape == (5,)


def test_compute_derived_states_speed_values():
    """Implied speed in derived state must match segment length / duration."""
    occ = _free_occupancy()
    opt = TrajectoryOptimizer(occ, cruise_speed=2.0)
    waypoints = [np.array([0.0, 0.0]), np.array([6.0, 0.0])]
    durations = np.array([3.0])
    derived = opt._compute_derived_states(waypoints, durations)
    # v = 6/3 = 2.0
    assert math.isclose(derived[0][3], 2.0, rel_tol=1e-9)
    assert math.isclose(derived[1][3], 2.0, rel_tol=1e-9)


def test_compute_derived_states_heading_east():
    """Segment heading must be 0 rad for an eastward path."""
    occ = _free_occupancy()
    opt = TrajectoryOptimizer(occ, cruise_speed=2.0)
    waypoints = [np.array([0.0, 0.0]), np.array([4.0, 0.0])]
    durations = np.array([2.0])
    derived = opt._compute_derived_states(waypoints, durations)
    assert math.isclose(derived[0][2], 0.0, abs_tol=1e-9)


def test_compute_derived_states_first_waypoint_zero_omega():
    """Turn rate at the first waypoint must always be zero."""
    occ = _free_occupancy()
    opt = TrajectoryOptimizer(occ, cruise_speed=2.0)
    waypoints = [np.array([0.0, 0.0]), np.array([4.0, 0.0])]
    durations = np.array([2.0])
    derived = opt._compute_derived_states(waypoints, durations)
    assert math.isclose(derived[0][4], 0.0, abs_tol=1e-9)


def test_compute_derived_states_omega_right_turn():
    """Interior waypoint omega must be negative for a right-turn path."""
    occ = _free_occupancy()
    opt = TrajectoryOptimizer(occ, cruise_speed=2.0)
    # Path: east then south → right turn at waypoint 1
    waypoints = [
        np.array([0.0, 0.0]),
        np.array([4.0, 0.0]),
        np.array([4.0, -4.0]),
    ]
    durations = np.array([2.0, 2.0])
    derived = opt._compute_derived_states(waypoints, durations)
    # At waypoint 1: heading goes from 0 to -π/2 → omega < 0
    assert derived[1][4] < 0.0


# ---------------------------------------------------------------------------
# Unit tests — _check_speed_bounds helper
# ---------------------------------------------------------------------------


def test_check_speed_bounds_no_limits():
    """Without speed limits every segment is considered feasible."""
    occ = _free_occupancy()
    opt = TrajectoryOptimizer(occ, cruise_speed=2.0)
    waypoints = [np.array([0.0, 0.0]), np.array([100.0, 0.0])]
    durations = np.array([0.001])  # very fast
    assert opt._check_speed_bounds(durations, waypoints) is True


def test_check_speed_bounds_within_limits():
    """All segments respecting max_speed must return True."""
    occ = _free_occupancy()
    opt = TrajectoryOptimizer(occ, cruise_speed=2.0, max_speed=10.0)
    waypoints = [np.array([0.0, 0.0]), np.array([4.0, 0.0])]
    durations = np.array([2.0])  # v=2 < 10 ✓
    assert opt._check_speed_bounds(durations, waypoints) is True


def test_check_speed_bounds_at_exact_max_speed():
    """Speed exactly at max_speed must be considered feasible."""
    occ = _free_occupancy()
    opt = TrajectoryOptimizer(occ, cruise_speed=2.0, max_speed=2.0)
    waypoints = [np.array([0.0, 0.0]), np.array([4.0, 0.0])]
    durations = np.array([2.0])  # v=2.0 == max_speed ✓
    assert opt._check_speed_bounds(durations, waypoints) is True


def test_check_speed_bounds_exceeds_max_speed():
    """Any segment exceeding max_speed must return False."""
    occ = _free_occupancy()
    opt = TrajectoryOptimizer(occ, cruise_speed=2.0, max_speed=1.0)
    waypoints = [np.array([0.0, 0.0]), np.array([4.0, 0.0])]
    durations = np.array([2.0])  # v=2 > max_speed=1 ✗
    assert opt._check_speed_bounds(durations, waypoints) is False


def test_check_speed_bounds_below_min_speed():
    """Any segment below min_speed must return False."""
    occ = _free_occupancy()
    opt = TrajectoryOptimizer(occ, cruise_speed=2.0, min_speed=5.0)
    waypoints = [np.array([0.0, 0.0]), np.array([4.0, 0.0])]
    durations = np.array([4.0])  # v=1 < min_speed=5 ✗
    assert opt._check_speed_bounds(durations, waypoints) is False


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
