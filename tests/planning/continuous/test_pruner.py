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

    Constructs a scenario where a greedy forward-scan from index 0 would
    commit a sub-optimal intermediate node, but the BFS finds a shorter
    subsequence.

    Path indices: 0, 1, 2, 3, 4
    Feasibility:
        0->1: OK  (consecutive)
        0->2: FAIL (blocked)
        0->3: OK  (non-consecutive long jump that greedy misses)
        0->4: FAIL
        1->2: OK  (consecutive)
        1->3: OK
        1->4: OK
        2->3: OK  (consecutive)
        2->4: OK
        3->4: OK  (consecutive)

    Greedy (from anchor 0):
        - try 0->2: FAIL → commit node 1, anchor=1
        - try 1->3: OK, 1->4: OK → best=4, done
        Greedy result: [0, 1, 4] → 3 nodes

    Optimal BFS:
        - From 0: 0->1 OK, 0->2 FAIL, 0->3 OK, 0->4 FAIL
        - BFS level 1 reached: {1, 3}
        - From 1: 1->2, 1->3, 1->4 reachable (3,4 new)
        - From 3: 3->4 reachable (4 already)
        - Shortest path to 4: 0->3->4 (2 hops) → [0, 3, 4] → 3 nodes (same)

    Hmm, same length in this case. Let me think of a better example.

    Path: 0, 1, 2, 3, 4, 5
    Feasibility (custom steer):
        consecutive always OK
        0->2: FAIL
        0->3: OK
        0->4: FAIL
        0->5: FAIL
        3->5: OK

    Greedy from anchor 0:
        - try 0->2: FAIL → commit 1, anchor=1
        - try 1->3, 1->4, 1->5: FAIL at some point
    Greedy result: 4+ nodes

    Optimal BFS:
        - 0->3: OK, then 3->5: OK → [0, 3, 5] → 3 nodes

    This is the canonical example where greedy is suboptimal.
    """
    occ = _free_occupancy()
    pruner = TrajectoryPruner(occ)

    path = [np.array([float(i), 0.0]) for i in range(6)]

    # feasible_pairs: only these edges are allowed (plus all consecutive).
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

    # Optimal: [0, 3, 5] — 3 nodes (2 hops).
    # Greedy would produce [0, 1, ...] because 0->2 fails.
    assert len(result) == 3
    assert np.allclose(result[0], path[0])
    assert np.allclose(result[1], path[3])
    assert np.allclose(result[2], path[5])


# ---------------------------------------------------------------------------
# Custom steer callable
# ---------------------------------------------------------------------------


def test_custom_steer_all_feasible():
    """With a custom steer that always returns True, all nodes are dropped."""
    occ = _free_occupancy()
    pruner = TrajectoryPruner(occ)
    path = _straight_path(8)

    def always_true(a, b):
        return True

    result = pruner.prune(path, steer=always_true)
    assert len(result) == 2


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
