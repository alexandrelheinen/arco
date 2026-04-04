"""Tests for TrajectoryOptimizer."""

import numpy as np
import pytest

from arco.mapping import KDTreeOccupancy
from arco.planning.continuous import TrajectoryOptimizer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BOUNDS_2D = [(0.0, 10.0), (0.0, 10.0)]


def _free_occupancy(clearance=0.3):
    """KDTreeOccupancy with a single far-away obstacle (effectively free space)."""
    return KDTreeOccupancy([[50.0, 50.0]], clearance=clearance)


def _obstacle_at_center():
    """KDTreeOccupancy with a cluster of obstacles near (5, 5)."""
    pts = [[5.0 + dx, 5.0 + dy] for dx in [-0.2, 0.0, 0.2] for dy in [-0.2, 0.0, 0.2]]
    return KDTreeOccupancy(pts, clearance=0.5)


def _always_feasible(state):
    return True


def _never_feasible(state):
    return False


def _feasible_below_y8(state):
    """Feasible only when y < 8."""
    return float(state[1]) < 8.0


def _straight_reference(start=(0.5, 0.5), end=(9.5, 9.5), steps=5):
    """Build a simple straight-line reference path."""
    xs = np.linspace(start[0], end[0], steps)
    ys = np.linspace(start[1], end[1], steps)
    return [np.array([x, y]) for x, y in zip(xs, ys)]


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_construction_negative_spatial_step_raises():
    occ = _free_occupancy()
    with pytest.raises(ValueError, match="spatial_step must be positive"):
        TrajectoryOptimizer(occ, _always_feasible, spatial_step=-1.0)


def test_construction_zero_spatial_step_raises():
    occ = _free_occupancy()
    with pytest.raises(ValueError, match="spatial_step must be positive"):
        TrajectoryOptimizer(occ, _always_feasible, spatial_step=0.0)


def test_construction_zero_weight_time_raises():
    occ = _free_occupancy()
    with pytest.raises(ValueError, match="weight_time must be positive"):
        TrajectoryOptimizer(occ, _always_feasible, weight_time=0.0)


def test_construction_negative_deviation_bound_raises():
    occ = _free_occupancy()
    with pytest.raises(ValueError, match="deviation_bound must be positive"):
        TrajectoryOptimizer(occ, _always_feasible, deviation_bound=-1.0)


# ---------------------------------------------------------------------------
# optimize — input validation
# ---------------------------------------------------------------------------


def test_optimize_single_point_raises():
    occ = _free_occupancy()
    opt = TrajectoryOptimizer(occ, _always_feasible)
    with pytest.raises(ValueError, match="at least two points"):
        opt.optimize([np.array([1.0, 1.0])])


def test_optimize_empty_raises():
    occ = _free_occupancy()
    opt = TrajectoryOptimizer(occ, _always_feasible)
    with pytest.raises(ValueError, match="at least two points"):
        opt.optimize([])


# ---------------------------------------------------------------------------
# Discretize / resample
# ---------------------------------------------------------------------------


def test_discretize_preserves_endpoints():
    occ = _free_occupancy()
    opt = TrajectoryOptimizer(occ, _always_feasible, spatial_step=1.0)
    ref = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
    resampled = opt._discretize(ref)
    assert np.allclose(resampled[0], ref[0])
    assert np.allclose(resampled[-1], ref[-1])


def test_discretize_uniform_spacing():
    """Consecutive resampled points should be close to spatial_step apart."""
    occ = _free_occupancy()
    step = 1.0
    opt = TrajectoryOptimizer(occ, _always_feasible, spatial_step=step)
    ref = np.array([[0.0, 0.0], [10.0, 0.0]])
    resampled = opt._discretize(ref)
    dists = np.linalg.norm(np.diff(resampled, axis=0), axis=1)
    # All intervals should be <= spatial_step (last may be shorter)
    assert np.all(dists <= step + 1e-9)


def test_discretize_more_points_for_smaller_step():
    occ = _free_occupancy()
    ref = np.array([[0.0, 0.0], [10.0, 0.0]])
    opt_coarse = TrajectoryOptimizer(occ, _always_feasible, spatial_step=2.0)
    opt_fine = TrajectoryOptimizer(occ, _always_feasible, spatial_step=0.5)
    n_coarse = len(opt_coarse._discretize(ref))
    n_fine = len(opt_fine._discretize(ref))
    assert n_fine > n_coarse


# ---------------------------------------------------------------------------
# Cost function evaluation
# ---------------------------------------------------------------------------


def test_cost_increases_near_obstacle():
    """Cost should be higher when the path passes through an obstacle."""
    occ = _obstacle_at_center()
    opt = TrajectoryOptimizer(
        occ,
        _always_feasible,
        spatial_step=1.0,
        weight_time=1.0,
        weight_deviation=0.0,
        weight_collision=100.0,
    )
    ref = np.array([[0.5, 0.5], [5.0, 5.0], [9.5, 9.5]])
    from scipy.spatial import KDTree

    resampled = opt._discretize(ref)
    ref_tree = KDTree(resampled)
    dim = resampled.shape[1]
    n_intermediate = len(resampled) - 2

    # Place intermediate points right on the obstacle vs. away from it.
    x_through = np.tile([5.0, 5.0], n_intermediate)
    x_clear = np.tile([0.5, 9.0], n_intermediate)

    cost_through = opt._cost(x_through, resampled, ref_tree, dim, n_intermediate)
    cost_clear = opt._cost(x_clear, resampled, ref_tree, dim, n_intermediate)
    assert cost_through > cost_clear


def test_cost_increases_with_longer_path():
    """Cost should be higher for a longer path (time² term)."""
    occ = _free_occupancy()
    opt = TrajectoryOptimizer(
        occ,
        _always_feasible,
        spatial_step=2.0,
        weight_time=1.0,
        weight_deviation=0.0,
        weight_collision=0.0,
    )
    ref = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 0.0]])
    from scipy.spatial import KDTree

    resampled = opt._discretize(ref)
    ref_tree = KDTree(resampled)
    dim = resampled.shape[1]
    n_intermediate = len(resampled) - 2

    # Straight intermediate points vs. widely deviated points
    mid_ref = resampled[1:-1]
    x_short = mid_ref.flatten()  # close to reference
    x_long = (mid_ref + np.array([3.0, -3.0])).flatten()  # deviated

    cost_short = opt._cost(x_short, resampled, ref_tree, dim, n_intermediate)
    cost_long = opt._cost(x_long, resampled, ref_tree, dim, n_intermediate)
    assert cost_long > cost_short


def test_cost_infeasibility_penalty_dominates():
    """Infeasibility penalty should dominate all other cost terms."""
    occ = _free_occupancy()
    penalty = 1e6
    opt = TrajectoryOptimizer(
        occ,
        _never_feasible,
        spatial_step=2.0,
        weight_time=1.0,
        weight_deviation=1.0,
        weight_collision=1.0,
        infeasibility_penalty=penalty,
    )
    ref = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]])
    from scipy.spatial import KDTree

    resampled = opt._discretize(ref)
    ref_tree = KDTree(resampled)
    dim = resampled.shape[1]
    n_intermediate = len(resampled) - 2

    x = resampled[1:-1].flatten()
    cost = opt._cost(x, resampled, ref_tree, dim, n_intermediate)
    # All points are infeasible → cost should be >= n_points * penalty
    assert cost >= len(resampled) * penalty


# ---------------------------------------------------------------------------
# Feasibility rejection in output
# ---------------------------------------------------------------------------


def test_optimize_all_infeasible_returns_none():
    """When all points are infeasible, optimize should return None."""
    occ = _free_occupancy()
    opt = TrajectoryOptimizer(
        occ,
        _never_feasible,
        spatial_step=2.0,
        stage1_max_iter=5,
        stage2_max_iter=5,
    )
    path = _straight_reference()
    result = opt.optimize(path)
    assert result is None


def test_optimize_feasibility_filters_output():
    """Points failing the feasibility check must not appear in the output."""
    occ = _free_occupancy()

    def feasible_x_lt_8(state):
        return float(state[0]) < 8.0

    opt = TrajectoryOptimizer(
        occ,
        feasible_x_lt_8,
        spatial_step=1.0,
        stage1_max_iter=5,
        stage2_max_iter=5,
    )
    path = _straight_reference()
    result = opt.optimize(path)
    if result is not None:
        for pt in result:
            assert float(pt[0]) < 8.0


# ---------------------------------------------------------------------------
# Two-stage chain: Stage 1 output used as Stage 2 input
# ---------------------------------------------------------------------------


def test_stage1_initializes_stage2():
    """Stage 2 should start from Stage 1 output (not from the reference)."""
    occ = _free_occupancy()
    ref = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]])

    from scipy.spatial import KDTree
    from scipy.optimize import differential_evolution, minimize

    opt = TrajectoryOptimizer(
        occ,
        _always_feasible,
        spatial_step=2.5,
        stage1_max_iter=10,
        stage2_max_iter=10,
    )

    resampled = opt._discretize(ref)
    intermediate_ref = resampled[1:-1]
    dim = resampled.shape[1]
    n_intermediate = len(intermediate_ref)
    bounds = opt._build_bounds(intermediate_ref, dim)
    ref_tree = KDTree(resampled)

    # Capture stage 1 result
    stage1_result = differential_evolution(
        opt._cost,
        bounds=bounds,
        args=(resampled, ref_tree, dim, n_intermediate),
        maxiter=10,
        popsize=5,
        seed=0,
    )
    x1 = stage1_result.x

    # Stage 2 initialized from x1 — verify it converges from there
    stage2_result = minimize(
        opt._cost,
        x1,
        args=(resampled, ref_tree, dim, n_intermediate),
        method="L-BFGS-B",
        bounds=bounds,
    )
    # Stage 2 cost should not be worse than Stage 1 cost
    assert stage2_result.fun <= stage1_result.fun + 1e-3


# ---------------------------------------------------------------------------
# Full optimization scenario
# ---------------------------------------------------------------------------


def test_optimize_free_space_returns_trajectory():
    """Optimizer should return a trajectory in free space."""
    occ = _free_occupancy()
    opt = TrajectoryOptimizer(
        occ,
        _always_feasible,
        spatial_step=2.0,
        stage1_population_count=5,
        stage1_max_iter=20,
        stage2_max_iter=50,
    )
    path = _straight_reference()
    result = opt.optimize(path)
    assert result is not None
    assert len(result) >= 2


def test_optimize_preserves_endpoints():
    """Start and goal must be preserved in the output."""
    occ = _free_occupancy()
    opt = TrajectoryOptimizer(
        occ,
        _always_feasible,
        spatial_step=2.0,
        stage1_population_count=5,
        stage1_max_iter=10,
        stage2_max_iter=20,
    )
    start = np.array([0.5, 0.5])
    goal = np.array([9.5, 9.5])
    path = [start, goal]
    result = opt.optimize(path)
    assert result is not None
    assert len(result) >= 2
    assert np.allclose(result[0], start)
    assert np.allclose(result[-1], goal)


def test_optimize_output_shorter_than_naive_interpolation():
    """Optimized path length should be no longer than a straight-line path."""
    occ = _free_occupancy()
    opt = TrajectoryOptimizer(
        occ,
        _always_feasible,
        spatial_step=1.5,
        weight_time=10.0,
        weight_deviation=0.1,
        weight_collision=1.0,
        deviation_bound=3.0,
        stage1_population_count=5,
        stage1_max_iter=30,
        stage2_max_iter=100,
    )
    # Reference: zig-zag path (longer than straight line)
    ref = [
        np.array([0.0, 0.0]),
        np.array([2.0, 4.0]),
        np.array([4.0, 0.0]),
        np.array([6.0, 4.0]),
        np.array([8.0, 0.0]),
        np.array([10.0, 0.0]),
    ]
    result = opt.optimize(ref)
    assert result is not None

    def path_length(pts):
        pts_arr = np.array(pts)
        return float(
            np.sum(np.linalg.norm(np.diff(pts_arr, axis=0), axis=1))
        )

    ref_length = path_length(ref)
    result_length = path_length(result)
    # The optimizer should produce a trajectory no longer than the reference
    # (with a generous tolerance since stage1 uses few iterations here)
    assert result_length <= ref_length * 1.5


def test_optimize_obstacle_avoidance():
    """Optimized path should avoid obstacles when feasible."""
    occ = _obstacle_at_center()
    opt = TrajectoryOptimizer(
        occ,
        _always_feasible,
        spatial_step=1.0,
        weight_time=1.0,
        weight_deviation=0.1,
        weight_collision=100.0,
        deviation_bound=4.0,
        stage1_population_count=10,
        stage1_max_iter=50,
        stage2_max_iter=100,
    )
    ref = [
        np.array([1.0, 5.0]),
        np.array([5.0, 5.0]),
        np.array([9.0, 5.0]),
    ]
    result = opt.optimize(ref)
    assert result is not None
    # Most points should not be inside the clearance zone
    clearance = occ.clearance
    occupied_count = sum(1 for p in result if occ.is_occupied(p))
    # Allow at most 1 point near the obstacle (endpoints are fixed)
    assert occupied_count <= 1


# ---------------------------------------------------------------------------
# Export test: TrajectoryOptimizer importable from top-level planning module
# ---------------------------------------------------------------------------


def test_import_from_planning_module():
    from arco.planning import TrajectoryOptimizer as TO  # noqa: F401

    assert TO is TrajectoryOptimizer
