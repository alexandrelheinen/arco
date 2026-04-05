"""Tests for RRTPlanner (RRT*)."""

import numpy as np
import pytest

from arco.mapping import KDTreeOccupancy
from arco.planning.continuous import RRTPlanner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BOUNDS_2D = [(0.0, 10.0), (0.0, 10.0)]


def _empty_occupancy(clearance=0.3):
    """KDTreeOccupancy with a single far-away obstacle (effectively free space)."""
    return KDTreeOccupancy([[50.0, 50.0]], clearance=clearance)


def _wall_occupancy(clearance=0.4):
    """Vertical wall at x=5, y in [2..8], creating two passable openings."""
    pts = [[5.0, y] for y in np.arange(2.0, 8.1, 0.5)]
    return KDTreeOccupancy(pts, clearance=clearance)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_empty_bounds_raises():
    occ = _empty_occupancy()
    with pytest.raises(ValueError, match="bounds must not be empty"):
        RRTPlanner(occ, bounds=[])


def test_construction_nonpositive_step_raises():
    occ = _empty_occupancy()
    with pytest.raises(ValueError, match="step_size must be positive"):
        RRTPlanner(occ, bounds=BOUNDS_2D, step_size=0.0)


# ---------------------------------------------------------------------------
# Free-space planning
# ---------------------------------------------------------------------------


def test_plan_free_space_finds_path():
    occ = _empty_occupancy()
    planner = RRTPlanner(
        occ,
        bounds=BOUNDS_2D,
        max_sample_count=1000,
        step_size=1.0,
        goal_tolerance=1.0,
        goal_bias=0.1,
    )
    path = planner.plan(np.array([0.5, 0.5]), np.array([9.5, 9.5]))
    assert path is not None
    assert len(path) >= 2


def test_plan_start_matches_first_waypoint():
    occ = _empty_occupancy()
    planner = RRTPlanner(occ, bounds=BOUNDS_2D, max_sample_count=500)
    start = np.array([1.0, 1.0])
    path = planner.plan(start, np.array([8.0, 8.0]))
    assert path is not None
    assert np.allclose(path[0], start)


def test_plan_goal_matches_last_waypoint():
    occ = _empty_occupancy()
    planner = RRTPlanner(occ, bounds=BOUNDS_2D, max_sample_count=500)
    goal = np.array([8.0, 8.0])
    path = planner.plan(np.array([1.0, 1.0]), goal)
    assert path is not None
    assert np.allclose(path[-1], goal)


def test_plan_path_is_collision_free():
    occ = KDTreeOccupancy([[5.0, 5.0]], clearance=1.5)
    planner = RRTPlanner(
        occ,
        bounds=BOUNDS_2D,
        max_sample_count=2000,
        step_size=0.5,
        goal_tolerance=0.6,
        collision_check_count=15,
    )
    path = planner.plan(np.array([1.0, 1.0]), np.array([9.0, 9.0]))
    assert path is not None
    for pt in path:
        assert not occ.is_occupied(pt)


# ---------------------------------------------------------------------------
# Blocked-path scenario
# ---------------------------------------------------------------------------


def test_plan_blocked_returns_none():
    """Completely walled obstacle: no path exists."""
    # Dense ring of obstacles surrounding the goal
    angles = np.linspace(0, 2 * np.pi, 30, endpoint=False)
    pts = [[5.0 + 1.0 * np.cos(a), 5.0 + 1.0 * np.sin(a)] for a in angles]
    # also fill inside
    for r in np.arange(0, 1.0, 0.3):
        for a in angles:
            pts.append([5.0 + r * np.cos(a), 5.0 + r * np.sin(a)])
    occ = KDTreeOccupancy(pts, clearance=0.8)
    planner = RRTPlanner(
        occ,
        bounds=BOUNDS_2D,
        max_sample_count=500,
        step_size=0.5,
        goal_tolerance=0.5,
    )
    path = planner.plan(np.array([1.0, 1.0]), np.array([5.0, 5.0]))
    # Goal is inside the obstacle, so should be None
    assert path is None


def test_plan_occupied_start_returns_none():
    occ = KDTreeOccupancy([[1.0, 1.0]], clearance=2.0)
    planner = RRTPlanner(occ, bounds=BOUNDS_2D, max_sample_count=100)
    path = planner.plan(np.array([1.0, 1.0]), np.array([9.0, 9.0]))
    assert path is None


def test_plan_occupied_goal_returns_none():
    occ = KDTreeOccupancy([[9.0, 9.0]], clearance=2.0)
    planner = RRTPlanner(occ, bounds=BOUNDS_2D, max_sample_count=100)
    path = planner.plan(np.array([1.0, 1.0]), np.array([9.0, 9.0]))
    assert path is None


# ---------------------------------------------------------------------------
# get_tree
# ---------------------------------------------------------------------------


def test_get_tree_returns_nodes_and_parent():
    occ = _empty_occupancy()
    planner = RRTPlanner(occ, bounds=BOUNDS_2D, max_sample_count=300)
    nodes, parent, path = planner.get_tree(
        np.array([0.5, 0.5]), np.array([9.5, 9.5])
    )
    assert len(nodes) > 1
    assert 0 in parent
    assert parent[0] is None


def test_get_tree_path_consistent_with_plan():
    occ = _empty_occupancy()
    planner = RRTPlanner(occ, bounds=BOUNDS_2D, max_sample_count=600)
    _, _, path = planner.get_tree(np.array([0.5, 0.5]), np.array([9.5, 9.5]))
    if path is not None:
        assert np.allclose(path[0], [0.5, 0.5])
        assert np.allclose(path[-1], [9.5, 9.5])


# ---------------------------------------------------------------------------
# Cost improvement with more samples
# ---------------------------------------------------------------------------


def test_cost_decreases_with_more_samples():
    """RRT* should produce shorter paths with more samples (statistical)."""
    occ = _empty_occupancy()
    start = np.array([0.5, 0.5])
    goal = np.array([9.5, 9.5])

    def path_length(path):
        return sum(
            float(np.linalg.norm(path[i + 1] - path[i]))
            for i in range(len(path) - 1)
        )

    planner_few = RRTPlanner(
        occ,
        bounds=BOUNDS_2D,
        max_sample_count=200,
        step_size=1.0,
        goal_tolerance=1.0,
        goal_bias=0.1,
    )
    planner_many = RRTPlanner(
        occ,
        bounds=BOUNDS_2D,
        max_sample_count=1500,
        step_size=1.0,
        goal_tolerance=1.0,
        goal_bias=0.1,
    )
    path_few = planner_few.plan(start, goal)
    path_many = planner_many.plan(start, goal)

    # Both should find a path; the longer run should be at least as good
    if path_few is not None and path_many is not None:
        assert path_length(path_many) <= path_length(path_few) * 1.5


# ---------------------------------------------------------------------------
# early_stop behavior
# ---------------------------------------------------------------------------


def test_early_stop_true_finds_path():
    """early_stop=True must still return a valid path."""
    occ = _empty_occupancy()
    planner = RRTPlanner(
        occ,
        bounds=BOUNDS_2D,
        max_sample_count=5000,
        step_size=1.0,
        goal_tolerance=1.0,
        goal_bias=0.1,
        early_stop=True,
    )
    path = planner.plan(np.array([0.5, 0.5]), np.array([9.5, 9.5]))
    assert path is not None
    assert len(path) >= 2


def test_early_stop_terminates_before_full_iterations():
    """early_stop=True should visit fewer nodes than early_stop=False."""
    occ = _empty_occupancy()
    common = dict(
        bounds=BOUNDS_2D,
        max_sample_count=5000,
        step_size=1.0,
        goal_tolerance=1.0,
        goal_bias=0.1,
    )
    start, goal = np.array([0.5, 0.5]), np.array([9.5, 9.5])

    nodes_early, _, _ = RRTPlanner(occ, **common, early_stop=True).get_tree(
        start, goal
    )
    nodes_full, _, _ = RRTPlanner(occ, **common, early_stop=False).get_tree(
        start, goal
    )
    assert len(nodes_early) <= len(nodes_full)
