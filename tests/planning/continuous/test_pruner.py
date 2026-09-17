"""Tests for TrajectoryPruner."""

import numpy as np
import pytest

from arco.mapping import KDTreeOccupancy
from arco.planning.continuous import TrajectoryPruner

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_STEP_2D = np.array([1.0, 1.0])
_STEP_3D = np.array([1.0, 1.0, 0.1])


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
    pruner = TrajectoryPruner(
        occ, step_size=_STEP_2D, collision_check_count=10
    )
    assert pruner.occupancy is occ
    assert pruner.collision_check_count == 10
    assert np.allclose(pruner.step_size, _STEP_2D)


def test_construction_stores_step_size_as_array():
    occ = _free_occupancy()
    pruner = TrajectoryPruner(occ, step_size=[0.5, 0.5])
    assert isinstance(pruner.step_size, np.ndarray)
    assert np.allclose(pruner.step_size, [0.5, 0.5])


def test_construction_default_collision_check_count():
    occ = _free_occupancy()
    pruner = TrajectoryPruner(occ, step_size=_STEP_2D)
    assert pruner.collision_check_count == 10


def test_construction_invalid_collision_check_count_raises():
    occ = _free_occupancy()
    with pytest.raises(ValueError, match="collision_check_count must be"):
        TrajectoryPruner(occ, step_size=_STEP_2D, collision_check_count=0)


def test_construction_negative_collision_check_count_raises():
    occ = _free_occupancy()
    with pytest.raises(ValueError, match="collision_check_count must be"):
        TrajectoryPruner(occ, step_size=_STEP_2D, collision_check_count=-1)


def test_construction_scalar_step_size_raises():
    """step_size must be a 1-D array; a scalar (0-D) is rejected."""
    occ = _free_occupancy()
    with pytest.raises(
        ValueError, match="step_size must be a non-empty 1-D array"
    ):
        TrajectoryPruner(occ, step_size=np.float64(1.0))


def test_construction_zero_step_size_raises():
    occ = _free_occupancy()
    with pytest.raises(
        ValueError, match="step_size elements must be strictly positive"
    ):
        TrajectoryPruner(occ, step_size=np.array([1.0, 0.0]))


def test_construction_negative_step_size_raises():
    occ = _free_occupancy()
    with pytest.raises(
        ValueError, match="step_size elements must be strictly positive"
    ):
        TrajectoryPruner(occ, step_size=np.array([1.0, -0.5]))


# ---------------------------------------------------------------------------
# Edge cases: empty and single-node paths
# ---------------------------------------------------------------------------


def test_prune_empty_path_returns_empty():
    occ = _free_occupancy()
    pruner = TrajectoryPruner(occ, step_size=_STEP_2D)
    result = pruner.prune([])
    assert result == []


def test_prune_single_node_path_returns_same():
    occ = _free_occupancy()
    pruner = TrajectoryPruner(occ, step_size=_STEP_2D)
    node = np.array([1.0, 2.0])
    result = pruner.prune([node])
    assert len(result) == 1
    assert np.allclose(result[0], node)


def test_prune_two_node_path_returns_both():
    occ = _free_occupancy()
    pruner = TrajectoryPruner(occ, step_size=_STEP_2D)
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
    pruner = TrajectoryPruner(occ, step_size=_STEP_2D)
    path = _straight_path(8)
    result = pruner.prune(path)
    assert np.allclose(result[0], path[0])


def test_pruned_path_ends_at_same_node():
    occ = _free_occupancy()
    pruner = TrajectoryPruner(occ, step_size=_STEP_2D)
    path = _straight_path(8)
    result = pruner.prune(path)
    assert np.allclose(result[-1], path[-1])


# ---------------------------------------------------------------------------
# Node count invariant
# ---------------------------------------------------------------------------


def test_pruned_node_count_le_original():
    occ = _free_occupancy()
    pruner = TrajectoryPruner(occ, step_size=_STEP_2D)
    path = _straight_path(10)
    result = pruner.prune(path)
    assert len(result) <= len(path)


# ---------------------------------------------------------------------------
# Maximum reduction: straight-line path in free space
# ---------------------------------------------------------------------------


def test_straight_path_pruned_to_two_nodes():
    """All intermediate nodes on a straight line in free space are skippable."""
    occ = _free_occupancy()
    pruner = TrajectoryPruner(occ, step_size=_STEP_2D)
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
    pruner = TrajectoryPruner(occ, step_size=_STEP_2D)

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
    pruner = TrajectoryPruner(occ, step_size=_STEP_2D)

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

    Path: 0, 1, 2, 3, 4, 5
    Feasibility (custom steer):
        consecutive always OK
        0->3: OK
        3->5: OK

    Optimal BFS:
        - 0->3: OK, then 3->5: OK → [0, 3, 5] → 3 nodes
    """
    occ = _free_occupancy()
    pruner = TrajectoryPruner(occ, step_size=_STEP_2D)

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
    pruner = TrajectoryPruner(occ, step_size=_STEP_2D)
    path = _straight_path(8)

    def always_true(a, b):
        return True

    result = pruner.prune(path, steer=always_true)
    assert len(result) == 2


def test_custom_steer_none_feasible():
    """With a custom steer that always returns False, all nodes are kept."""
    occ = _free_occupancy()
    pruner = TrajectoryPruner(occ, step_size=_STEP_2D)
    path = _straight_path(6)

    def always_false(a, b):
        return False

    result = pruner.prune(path, steer=always_false)
    assert len(result) == len(path)


def test_custom_steer_called_with_correct_nodes():
    """Verify the steer callable receives the correct node pairs."""
    occ = _free_occupancy()
    pruner = TrajectoryPruner(occ, step_size=_STEP_2D)
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
    pruner = TrajectoryPruner(occ, step_size=_STEP_2D)
    path = [np.array([float(i), 0.0]) for i in range(10)]
    result = pruner.prune(path)

    original_as_tuples = {tuple(p) for p in path}
    for node in result:
        assert tuple(node) in original_as_tuples


# ---------------------------------------------------------------------------
# Internal helper: _segment_free — basic correctness
# ---------------------------------------------------------------------------










# ---------------------------------------------------------------------------
# Step-size formula: n computation — 2-D case
# ---------------------------------------------------------------------------








# ---------------------------------------------------------------------------
# Step-size formula: n computation — 3-D case
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# Step-size formula: D[i] = 0 for some dimension
# ---------------------------------------------------------------------------








# ---------------------------------------------------------------------------
# Under-sampled primitive: step-size criterion prevents obstacle crossing
# ---------------------------------------------------------------------------


