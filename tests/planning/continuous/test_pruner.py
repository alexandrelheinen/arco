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


def test_pruned_path_always_includes_prelast_node():
    """path[-2] must always appear in the pruned result.

    The pre-last node (path[-2]) is the last tree node placed by the planner
    within goal_tolerance of the goal.  Preserving it prevents the pruner from
    replacing the final tree-to-goal segment with a direct shortcut that might
    clip an obstacle.
    """
    occ = _free_occupancy()
    pruner = TrajectoryPruner(occ)
    path = _straight_path(8)
    result = pruner.prune(path)
    assert np.allclose(result[-2], path[-2])


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


def test_straight_path_pruned_to_three_nodes():
    """Intermediate nodes on a straight line are skipped but pre-last kept.

    The pruner always preserves path[-2] (the pre-last node) to protect the
    final approach segment from being replaced by a direct-to-goal shortcut
    that might clip an obstacle.  On a 10-node collinear path the result is
    therefore [start, path[-2], goal] — three nodes.
    """
    occ = _free_occupancy()
    pruner = TrajectoryPruner(occ)
    path = _straight_path(10)  # 10 collinear nodes
    result = pruner.prune(path)
    # Expect start + pre-last + goal (3 nodes).
    assert len(result) == 3
    assert np.allclose(result[0], path[0])
    assert np.allclose(result[-2], path[-2])
    assert np.allclose(result[-1], path[-1])


# ---------------------------------------------------------------------------
# No reduction: path with obstacles blocking all shortcuts
# ---------------------------------------------------------------------------


def test_no_reduction_when_all_shortcuts_blocked():
    """When every shortcut is blocked, the pruned path equals the original.

    Uses a custom steer that only allows consecutive-pair connections to
    ensure the scenario is deterministic and independent of obstacle geometry.
    """
    occ = _free_occupancy()
    pruner = TrajectoryPruner(occ)

    path = [np.array([float(i), 0.0]) for i in range(6)]

    # steer only allows jumps of exactly 1 (consecutive pairs).
    def consecutive_only(a, b):
        for i, node in enumerate(path):
            if np.allclose(a, node):
                for j, node2 in enumerate(path):
                    if np.allclose(b, node2):
                        return abs(i - j) == 1
        return False

    result = pruner.prune(path, steer=consecutive_only)
    assert len(result) == len(path)


# ---------------------------------------------------------------------------
# Mixed path: some nodes skippable, some not
# ---------------------------------------------------------------------------


def test_mixed_path_partial_reduction():
    """Path where only some intermediate nodes can be skipped.

    Uses a custom steer that blocks specific edges to create a deterministic
    mixed scenario.
    """
    occ = _free_occupancy()
    pruner = TrajectoryPruner(occ)

    # path: 0-1-2-3-4-5-6 (indices)
    # steer: consecutive always OK, skip across index 3 is blocked,
    #        collinear runs before and after index 3 are skippable.
    path = [np.array([float(i), 0.0]) for i in range(7)]
    blocked_pairs = {(0, 3), (1, 3), (3, 6), (3, 5)}

    def partial_steer(a, b):
        for i, node in enumerate(path):
            if np.allclose(a, node):
                for j, node2 in enumerate(path):
                    if np.allclose(b, node2):
                        if (i, j) in blocked_pairs:
                            return False
                        return True
        return True

    result = pruner.prune(path, steer=partial_steer)
    assert len(result) < len(path)
    assert np.allclose(result[0], path[0])
    assert np.allclose(result[-1], path[-1])


# ---------------------------------------------------------------------------
# Optimality: BFS finds fewer nodes than a greedy forward scan would
# ---------------------------------------------------------------------------


def test_bfs_finds_optimal_solution_over_greedy():
    """BFS-based pruner finds the minimum-waypoint path.

    Constructs a scenario where a greedy forward-scan would commit a
    sub-optimal intermediate node, but the BFS finds a shorter subsequence.

    Path: 0, 1, 2, 3, 4, 5  (indices)
    Feasibility (custom steer):
        consecutive always OK
        0->2: FAIL
        0->3: OK
        0->4: FAIL
        0->5: FAIL
        3->5: OK  (but 5 is path[-1] = exact goal, not in inner BFS)

    The pruner runs BFS over the *inner* path 0..4 (path[-2] = index 4),
    then appends path[5] unconditionally.

    Inner BFS (indices 0..4):
        From 0: 0->1 OK, 0->2 FAIL, 0->3 OK, 0->4 FAIL
        From 1 (level-1): 1->2, 1->3, 1->4 → 2,3 already seen; 4 new
        From 3 (level-1): 3->4 → already queued
        Shortest path to inner goal (4): 0->3->4 (2 hops)
    Append path[5] → result: [path[0], path[3], path[4], path[5]] — 4 nodes.

    Greedy (anchor 0):
        try 0->2: FAIL → commit 1, anchor=1
        try 1->3: OK, 1->4: OK → inner result [0, 1, 3, 4]
        append path[5] → [path[0], path[1], path[3], path[4], path[5]] — 5 nodes.

    BFS (4 nodes) beats greedy (5 nodes).

    Note: the (3, 5) edge in feasible_pairs is included for completeness but
    is not exercised by the inner BFS — path[5] is always appended
    unconditionally after the inner path is reconstructed.
    """
    occ = _free_occupancy()
    pruner = TrajectoryPruner(occ)

    path = [np.array([float(i), 0.0]) for i in range(6)]

    # feasible_pairs: only these edges are allowed (plus all consecutive).
    # (3, 5) is defined but irrelevant: path[5] is the exact goal and is
    # always appended unconditionally by the pruner, not reached via BFS.
    feasible_pairs = {
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (0, 3),
        (3, 5),
    }

    def custom_steer(a, b):
        for i, node in enumerate(path):
            if np.allclose(a, node):
                for j, node2 in enumerate(path):
                    if np.allclose(b, node2):
                        return (i, j) in feasible_pairs
        return False

    result = pruner.prune(path, steer=custom_steer)

    # Optimal inner path: [0, 3, 4], then exact goal appended → 4 nodes.
    # Greedy would produce [0, 1, 3, 4, 5] — 5 nodes.
    assert len(result) == 4
    assert np.allclose(result[0], path[0])
    assert np.allclose(result[1], path[3])
    assert np.allclose(result[2], path[4])  # pre-last always included
    assert np.allclose(result[3], path[5])  # exact goal always appended


# ---------------------------------------------------------------------------
# Custom steer callable
# ---------------------------------------------------------------------------


def test_custom_steer_all_feasible():
    """With all-feasible steer, only intermediate nodes (not pre-last) dropped."""
    occ = _free_occupancy()
    pruner = TrajectoryPruner(occ)
    path = _straight_path(8)

    def always_true(a, b):
        return True

    result = pruner.prune(path, steer=always_true)
    # Pre-last node (path[-2]) is always kept, so result is [start, path[-2], goal].
    assert len(result) == 3
    assert np.allclose(result[0], path[0])
    assert np.allclose(result[-2], path[-2])
    assert np.allclose(result[-1], path[-1])


def test_custom_steer_none_feasible():
    """With a custom steer that always returns False, all nodes are kept."""
    occ = _free_occupancy()
    pruner = TrajectoryPruner(occ)
    path = _straight_path(6)

    def always_false(a, b):
        return False

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
        return False  # block all shortcuts so BFS falls back to original

    pruner.prune(path, steer=recording_steer)

    # BFS explores from index 0: first call is path[0] -> path[1].
    assert len(calls) >= 1
    assert np.allclose(calls[0][0], path[0])
    assert np.allclose(calls[0][1], path[1])


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


def test_long_segment_detects_obstacle_with_adaptive_sampling():
    """Adaptive density catches an obstacle that a fixed count would miss.

    A fixed collision_check_count=2 places only 4 samples, each ~16.7 m
    apart on a 50 m segment, entirely missing an obstacle at the 25 m
    mark.  Adaptive sampling enforces spacing ≤ clearance/2, guaranteeing
    the obstacle is detected.
    """
    # Obstacle exactly at the midpoint of the segment.
    occ = KDTreeOccupancy([[25.0, 0.0]], clearance=0.5)
    pruner = TrajectoryPruner(occ, collision_check_count=2)
    a = np.array([0.0, 0.0])
    b = np.array([50.0, 0.0])
    # With adaptive sampling the obstacle is detected → segment is not free.
    assert pruner._segment_free(a, b) is False


def test_long_free_segment_returns_true():
    """Long segment that avoids all obstacles is correctly classified free."""
    # Obstacle is 25 m off the path (perpendicular), never within clearance.
    occ = KDTreeOccupancy([[0.0, 25.0]], clearance=0.5)
    pruner = TrajectoryPruner(occ, collision_check_count=2)
    a = np.array([0.0, 0.0])
    b = np.array([50.0, 0.0])
    assert pruner._segment_free(a, b) is True
