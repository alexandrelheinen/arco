"""Tests for SSTPlanner."""

import numpy as np
import pytest

from arco.mapping import KDTreeOccupancy
from arco.planning.continuous import SSTPlanner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BOUNDS_2D = [(0.0, 10.0), (0.0, 10.0)]


def _empty_occupancy(clearance=0.3):
    return KDTreeOccupancy([[50.0, 50.0]], clearance=clearance)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_empty_bounds_raises():
    occ = _empty_occupancy()
    with pytest.raises(ValueError, match="bounds must not be empty"):
        SSTPlanner(occ, bounds=[])


def test_construction_nonpositive_step_raises():
    occ = _empty_occupancy()
    with pytest.raises(ValueError, match="step_size must be positive"):
        SSTPlanner(occ, bounds=BOUNDS_2D, step_size=0.0)


# ---------------------------------------------------------------------------
# Free-space planning
# ---------------------------------------------------------------------------


def test_plan_free_space_finds_path():
    occ = _empty_occupancy()
    planner = SSTPlanner(
        occ,
        bounds=BOUNDS_2D,
        max_sample_count=2000,
        step_size=1.0,
        goal_tolerance=1.0,
        witness_radius=0.5,
        goal_bias=0.1,
    )
    path = planner.plan(np.array([0.5, 0.5]), np.array([9.5, 9.5]))
    assert path is not None
    assert len(path) >= 2


def test_plan_start_matches_first_waypoint():
    occ = _empty_occupancy()
    planner = SSTPlanner(occ, bounds=BOUNDS_2D, max_sample_count=1000)
    start = np.array([1.0, 1.0])
    path = planner.plan(start, np.array([8.0, 8.0]))
    assert path is not None
    assert np.allclose(path[0], start)


def test_plan_goal_matches_last_waypoint():
    occ = _empty_occupancy()
    planner = SSTPlanner(occ, bounds=BOUNDS_2D, max_sample_count=1000)
    goal = np.array([8.0, 8.0])
    path = planner.plan(np.array([1.0, 1.0]), goal)
    assert path is not None
    assert np.allclose(path[-1], goal)


def test_plan_path_is_collision_free():
    occ = KDTreeOccupancy([[5.0, 5.0]], clearance=1.5)
    planner = SSTPlanner(
        occ,
        bounds=BOUNDS_2D,
        max_sample_count=3000,
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


def test_plan_occupied_start_returns_none():
    occ = KDTreeOccupancy([[1.0, 1.0]], clearance=2.0)
    planner = SSTPlanner(occ, bounds=BOUNDS_2D, max_sample_count=100)
    path = planner.plan(np.array([1.0, 1.0]), np.array([9.0, 9.0]))
    assert path is None


def test_plan_occupied_goal_returns_none():
    occ = KDTreeOccupancy([[9.0, 9.0]], clearance=2.0)
    planner = SSTPlanner(occ, bounds=BOUNDS_2D, max_sample_count=100)
    path = planner.plan(np.array([1.0, 1.0]), np.array([9.0, 9.0]))
    assert path is None


# ---------------------------------------------------------------------------
# get_tree
# ---------------------------------------------------------------------------


def test_get_tree_returns_nodes_and_parent():
    occ = _empty_occupancy()
    planner = SSTPlanner(occ, bounds=BOUNDS_2D, max_sample_count=500)
    nodes, parent, path = planner.get_tree(
        np.array([0.5, 0.5]), np.array([9.5, 9.5])
    )
    assert len(nodes) >= 1
    assert 0 in parent


def test_get_tree_sparser_than_rrt():
    """SST active tree should stay compact compared to RRT*."""
    from arco.planning.continuous import RRTPlanner

    occ = _empty_occupancy()
    sst = SSTPlanner(
        occ,
        bounds=BOUNDS_2D,
        max_sample_count=1000,
        step_size=1.0,
        witness_radius=1.5,
    )
    rrt = RRTPlanner(
        occ,
        bounds=BOUNDS_2D,
        max_sample_count=1000,
        step_size=1.0,
    )
    sst_nodes, _, _ = sst.get_tree(np.array([0.5, 0.5]), np.array([9.5, 9.5]))
    rrt_nodes, _, _ = rrt.get_tree(np.array([0.5, 0.5]), np.array([9.5, 9.5]))
    # SST should have fewer active nodes than RRT* total nodes
    assert len(sst_nodes) <= len(rrt_nodes)


# ---------------------------------------------------------------------------
# early_stop behavior
# ---------------------------------------------------------------------------


def test_early_stop_true_finds_path():
    """early_stop=True must still return a valid path."""
    occ = _empty_occupancy()
    planner = SSTPlanner(
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
        witness_radius=0.5,
    )
    start, goal = np.array([0.5, 0.5]), np.array([9.5, 9.5])

    nodes_early, _, _ = SSTPlanner(occ, **common, early_stop=True).get_tree(
        start, goal
    )
    nodes_full, _, _ = SSTPlanner(occ, **common, early_stop=False).get_tree(
        start, goal
    )
    assert len(nodes_early) <= len(nodes_full)
