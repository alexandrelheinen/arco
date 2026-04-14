"""Tests for TrajectoryPruner."""

import numpy as np
import pytest

from arco.mapping import KDTreeOccupancy
from arco.planning.continuous import TrajectoryPruner

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _free_occupancy(clearance=0.1):
    """Effectively free-space occupancy (single far obstacle)."""
    return KDTreeOccupancy([[1000.0, 1000.0]], clearance=clearance)


def _wall_occupancy(wall_x=5.0, clearance=0.5):
    """Vertical wall at x=wall_x blocking direct horizontal crossings."""
    pts = [[wall_x, y] for y in np.arange(0.0, 11.0, 0.3)]
    return KDTreeOccupancy(pts, clearance=clearance)


def _straight_path(node_count=6):
    """Horizontal path with node_count nodes along y=0."""
    return [np.array([float(i), 0.0]) for i in range(node_count)]


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_construction_valid():
    occ = _free_occupancy()
    pruner = TrajectoryPruner(occ, collision_check_count=10)
    assert pruner.occupancy is occ
    assert pruner.collision_check_count == 10


def test_construction_default_collision_check_count():
    occ = _free_occupancy()
    pruner = TrajectoryPruner(occ)
    assert pruner.collision_check_count == 10


def test_construction_invalid_collision_check_count_raises():
    occ = _free_occupancy()
    with pytest.raises(ValueError, match="collision_check_count must be"):
        TrajectoryPruner(occ, collision_check_count=0)


def test_construction_negative_collision_check_count_raises():
    occ = _free_occupancy()
    with pytest.raises(ValueError, match="collision_check_count must be"):
        TrajectoryPruner(occ, collision_check_count=-1)


# ---------------------------------------------------------------------------
# Edge cases: empty and single-node paths
# ---------------------------------------------------------------------------


def test_prune_empty_path_returns_empty():
    occ = _free_occupancy()
    pruner = TrajectoryPruner(occ)
    result = pruner.prune([])
    assert result == []


def test_prune_single_node_path_returns_same():
    occ = _free_occupancy()
    pruner = TrajectoryPruner(occ)
    node = np.array([1.0, 2.0])
    result = pruner.prune([node])
    assert len(result) == 1
    assert np.allclose(result[0], node)


def test_prune_two_node_path_returns_both():
    occ = _free_occupancy()
    pruner = TrajectoryPruner(occ)
    path = [np.array([0.0, 0.0]), np.array([1.0, 1.0])]
    result = pruner.prune(path)
    assert len(result) == 2
    assert np.allclose(result[0], path[0])
    assert np.allclose(result[-1], path[-1])


# ---------------------------------------------------------------------------
# Start and end preservation
# ---------------------------------------------------------------------------


def test_pruned_path_starts_at_same_node():
    occ = _free_occupancy()
    pruner = TrajectoryPruner(occ)
    path = _straight_path(8)
    result = pruner.prune(path)
    assert np.allclose(result[0], path[0])


def test_pruned_path_ends_at_same_node():
    occ = _free_occupancy()
    pruner = TrajectoryPruner(occ)
    path = _straight_path(8)
    result = pruner.prune(path)
    assert np.allclose(result[-1], path[-1])


# ---------------------------------------------------------------------------
# Node count invariant
# ---------------------------------------------------------------------------


def test_pruned_node_count_le_original():
    occ = _free_occupancy()
    pruner = TrajectoryPruner(occ)
    path = _straight_path(10)
    result = pruner.prune(path)
    assert len(result) <= len(path)


# ---------------------------------------------------------------------------
# Maximum reduction: straight-line path in free space
# ---------------------------------------------------------------------------


def test_straight_path_pruned_to_two_nodes():
    """All intermediate nodes on a straight line in free space are skippable."""
    occ = _free_occupancy()
    pruner = TrajectoryPruner(occ)
    path = _straight_path(10)  # 10 collinear nodes
    result = pruner.prune(path)
    # Expect only start + goal since every direct connection is free.
    assert len(result) == 2
    assert np.allclose(result[0], path[0])
    assert np.allclose(result[-1], path[-1])


# ---------------------------------------------------------------------------
# No reduction: path with obstacles blocking all shortcuts
# ---------------------------------------------------------------------------


def test_no_reduction_when_all_shortcuts_blocked():
    """When every shortcut is blocked, the pruned path equals the original."""
    # Build a zigzag path that crosses the wall at each step, but
    # alternates sides so consecutive nodes are always on opposite sides.
    # Direct skip of a node would cross the wall.
    wall_x = 5.0
    occ = _wall_occupancy(wall_x=wall_x, clearance=0.6)
    pruner = TrajectoryPruner(occ, collision_check_count=20)

    # Path strictly on one side — build a path where consecutive pairs
    # are connected but non-adjacent nodes cross the wall.
    # Simple approach: two separate segments that alternate across the wall.
    # Instead, use a path of nodes at x=4, x=6, x=4, x=6, ...
    # Consecutive edges cross the wall (OK, because the planner guarantees
    # them), but every skip would cross the wall twice and be blocked.
    path = []
    for i in range(6):
        x = wall_x - 0.5 if i % 2 == 0 else wall_x + 0.5
        path.append(np.array([x, float(i)]))

    result = pruner.prune(path)
    # The pruned path must still contain at least as many nodes as strictly
    # required — in this case, no skip is possible, so all nodes are kept.
    assert len(result) == len(path)


# ---------------------------------------------------------------------------
# Mixed path: some nodes skippable, some not
# ---------------------------------------------------------------------------


def test_mixed_path_partial_reduction():
    """Path where only some intermediate nodes can be skipped."""
    # Straight section followed by a wall crossing, then straight again.
    wall_x = 5.0
    occ = _wall_occupancy(wall_x=wall_x, clearance=0.6)
    pruner = TrajectoryPruner(occ, collision_check_count=20)

    # Nodes on the left side (x<5): collinear, all skippable.
    left_nodes = [np.array([float(i), 0.0]) for i in range(1, 5)]
    # One crossing edge through the wall opening (y>10.5 is clear of wall).
    cross_node = np.array([wall_x, 11.0])
    # Nodes on the right side: collinear, all skippable.
    right_nodes = [np.array([float(i), 11.0]) for i in range(6, 10)]

    path = left_nodes + [cross_node] + right_nodes
    result = pruner.prune(path)

    # The result must have fewer nodes than the original (collinear sections
    # collapsed) but the start and end are preserved.
    assert len(result) < len(path)
    assert np.allclose(result[0], path[0])
    assert np.allclose(result[-1], path[-1])


# ---------------------------------------------------------------------------
# Custom steer callable
# ---------------------------------------------------------------------------


def test_custom_steer_all_feasible():
    """With a custom steer that always returns True, all nodes are dropped."""
    occ = _free_occupancy()
    pruner = TrajectoryPruner(occ)
    path = _straight_path(8)

    always_true = lambda a, b: True  # noqa: E731
    result = pruner.prune(path, steer=always_true)
    assert len(result) == 2


def test_custom_steer_none_feasible():
    """With a custom steer that always returns False, all nodes are kept."""
    occ = _free_occupancy()
    pruner = TrajectoryPruner(occ)
    path = _straight_path(6)

    always_false = lambda a, b: False  # noqa: E731
    result = pruner.prune(path, steer=always_false)
    assert len(result) == len(path)


def test_custom_steer_called_with_correct_nodes():
    """Verify the steer callable receives the correct node pairs."""
    occ = _free_occupancy()
    pruner = TrajectoryPruner(occ)
    path = [np.array([float(i), 0.0]) for i in range(4)]
    calls = []

    def recording_steer(a, b):
        calls.append((a.copy(), b.copy()))
        return False  # block all shortcuts so all nodes are kept

    pruner.prune(path, steer=recording_steer)

    # The steer should have been called with path[0] → path[2] at least.
    assert len(calls) >= 1
    assert np.allclose(calls[0][0], path[0])
    assert np.allclose(calls[0][1], path[2])


# ---------------------------------------------------------------------------
# Result is a strict subset of the input nodes
# ---------------------------------------------------------------------------


def test_pruned_nodes_are_subset_of_original():
    """Every node in the pruned path must appear in the original path."""
    occ = _free_occupancy()
    pruner = TrajectoryPruner(occ)
    path = [np.array([float(i), 0.0]) for i in range(10)]
    result = pruner.prune(path)

    original_as_tuples = {tuple(p) for p in path}
    for node in result:
        assert tuple(node) in original_as_tuples


# ---------------------------------------------------------------------------
# Internal helper: _segment_free
# ---------------------------------------------------------------------------


def test_segment_free_in_free_space():
    occ = _free_occupancy()
    pruner = TrajectoryPruner(occ)
    a = np.array([0.0, 0.0])
    b = np.array([5.0, 5.0])
    assert pruner._segment_free(a, b) is True


def test_segment_free_through_obstacle():
    occ = KDTreeOccupancy([[2.5, 2.5]], clearance=0.5)
    pruner = TrajectoryPruner(occ, collision_check_count=20)
    a = np.array([0.0, 0.0])
    b = np.array([5.0, 5.0])
    assert pruner._segment_free(a, b) is False
