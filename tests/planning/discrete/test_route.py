"""Unit tests for RouteRouter."""

from __future__ import annotations

import math

import pytest

from arco.mapping.graph import WeightedGraph
from arco.planning.discrete import RouteRouter


def _grid_graph() -> WeightedGraph:
    """Return a simple 3×3 grid graph for testing.

    Layout:
        0 - 1 - 2
        |   |   |
        3 - 4 - 5
        |   |   |
        6 - 7 - 8

    Each node is 10 units apart.
    """
    g = WeightedGraph()
    for i in range(3):
        for j in range(3):
            node_id = i * 3 + j
            g.add_node(node_id, float(j * 10), float(i * 10))

    # Horizontal edges
    for i in range(3):
        for j in range(2):
            node_a = i * 3 + j
            node_b = i * 3 + j + 1
            g.add_edge(node_a, node_b)

    # Vertical edges
    for i in range(2):
        for j in range(3):
            node_a = i * 3 + j
            node_b = (i + 1) * 3 + j
            g.add_edge(node_a, node_b)

    return g


class TestRouteRouterBasic:
    """Test basic RouteRouter functionality."""

    def test_plan_exact_node_positions(self):
        """Route from one node to another when query positions are exact."""
        g = _grid_graph()
        router = RouteRouter(g)
        result = router.plan(0.0, 0.0, 20.0, 20.0)

        assert result is not None
        assert result.start_node == 0
        assert result.goal_node == 8
        assert result.path[0] == 0
        assert result.path[-1] == 8
        assert result.start_distance == 0.0
        assert result.goal_distance == 0.0

    def test_plan_nearby_positions(self):
        """Route from positions near nodes."""
        g = _grid_graph()
        router = RouteRouter(g)
        # Start near node 0, goal near node 8
        result = router.plan(0.5, 0.5, 19.5, 19.5)

        assert result is not None
        assert result.start_node == 0
        assert result.goal_node == 8
        assert math.isclose(
            result.start_distance, math.hypot(0.5, 0.5), abs_tol=1e-6
        )
        assert math.isclose(
            result.goal_distance, math.hypot(0.5, 0.5), abs_tol=1e-6
        )

    def test_plan_same_start_goal(self):
        """Route from a position to itself."""
        g = _grid_graph()
        router = RouteRouter(g)
        result = router.plan(10.0, 10.0, 10.1, 10.1)

        assert result is not None
        assert result.start_node == result.goal_node == 4
        # Path should be single node
        assert result.path == [4]

    def test_plan_disconnected_graph(self):
        """Route fails on disconnected graph."""
        g = WeightedGraph()
        g.add_node(0, 0.0, 0.0)
        g.add_node(1, 100.0, 100.0)
        # No edges – disconnected

        router = RouteRouter(g)
        result = router.plan(0.0, 0.0, 100.0, 100.0)
        assert result is None

    def test_plan_path_length(self):
        """Route path length is reasonable."""
        g = _grid_graph()
        router = RouteRouter(g)
        result = router.plan(0.0, 0.0, 20.0, 0.0)

        assert result is not None
        # Path from node 0 to node 2 should be [0, 1, 2]
        assert len(result.path) == 3
        assert result.path == [0, 1, 2]


class TestRouteRouterActivationRadius:
    """Test activation radius constraints."""

    def test_start_outside_radius_fails(self):
        """Route fails when start is outside activation radius."""
        g = _grid_graph()
        router = RouteRouter(g, activation_radius=5.0)
        # Start at (50, 50) – far from all nodes
        result = router.plan(50.0, 50.0, 10.0, 10.0)
        assert result is None

    def test_goal_outside_radius_fails(self):
        """Route fails when goal is outside activation radius."""
        g = _grid_graph()
        router = RouteRouter(g, activation_radius=5.0)
        # Goal at (50, 50) – far from all nodes
        result = router.plan(10.0, 10.0, 50.0, 50.0)
        assert result is None

    def test_both_within_radius_succeeds(self):
        """Route succeeds when both start and goal are within radius."""
        g = _grid_graph()
        router = RouteRouter(g, activation_radius=5.0)
        # Both positions within 5 units of nearest nodes
        result = router.plan(2.0, 2.0, 18.0, 18.0)
        assert result is not None
        assert result.start_node == 0
        assert result.goal_node == 8

    def test_no_radius_allows_any_distance(self):
        """Route succeeds at any distance when radius is None."""
        g = _grid_graph()
        router = RouteRouter(g, activation_radius=None)
        # Very far positions should still work
        result = router.plan(-100.0, -100.0, 100.0, 100.0)
        assert result is not None
        # Should snap to corner nodes
        assert result.start_node == 0
        assert result.goal_node == 8


class TestRouteRouterProjection:
    """Test projection metadata in results."""

    def test_projection_coordinates(self):
        """Result includes projection coordinates."""
        g = _grid_graph()
        router = RouteRouter(g)
        result = router.plan(5.0, 5.0, 15.0, 15.0)

        assert result is not None
        # Start at (5, 5) is closest to node 4 at (10, 10), but (0,0) is also close
        # Let's check it projects to the nearest node
        start_node_pos = g.position(result.start_node)
        assert result.start_projection == start_node_pos

        goal_node_pos = g.position(result.goal_node)
        assert result.goal_projection == goal_node_pos

    def test_projection_distances(self):
        """Result includes projection distances."""
        g = _grid_graph()
        router = RouteRouter(g)
        result = router.plan(1.0, 2.0, 19.0, 18.0)

        assert result is not None
        # Start at (1, 2) projects to node 0 at (0, 0)
        expected_start_dist = math.hypot(1.0, 2.0)
        assert math.isclose(
            result.start_distance, expected_start_dist, abs_tol=1e-6
        )

        # Goal at (19, 18) projects to node 8 at (20, 20)
        expected_goal_dist = math.hypot(1.0, 2.0)
        assert math.isclose(
            result.goal_distance, expected_goal_dist, abs_tol=1e-6
        )


class TestRouteRouterEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_graph(self):
        """Route fails on empty graph."""
        g = WeightedGraph()
        router = RouteRouter(g)
        result = router.plan(0.0, 0.0, 10.0, 10.0)
        assert result is None

    def test_single_node_graph(self):
        """Route succeeds on single-node graph if start and goal snap to it."""
        g = WeightedGraph()
        g.add_node(0, 5.0, 5.0)
        router = RouteRouter(g)
        result = router.plan(4.0, 4.0, 6.0, 6.0)

        assert result is not None
        assert result.start_node == 0
        assert result.goal_node == 0
        assert result.path == [0]

    def test_optimal_path_selection(self):
        """Route selects the optimal (shortest) path."""
        g = WeightedGraph()
        # Diamond graph with shortcut
        #     1
        #    / \
        #   0   3
        #    \ /
        #     2
        g.add_node(0, 0.0, 0.0)
        g.add_node(1, 1.0, 1.0)
        g.add_node(2, 1.0, -1.0)
        g.add_node(3, 2.0, 0.0)
        g.add_edge(0, 1, weight=2.0)
        g.add_edge(1, 3, weight=2.0)
        g.add_edge(0, 2, weight=2.0)
        g.add_edge(2, 3, weight=2.0)
        g.add_edge(0, 3, weight=1.0)  # Direct shortcut

        router = RouteRouter(g)
        result = router.plan(0.0, 0.0, 2.0, 0.0)

        assert result is not None
        # Shortest path should be direct: [0, 3]
        assert result.path == [0, 3]


class TestRouteRouterDocumentedBenchmarks:
    """Test documented benchmark scenarios for acceptance criteria."""

    def test_benchmark_straight_road(self):
        """Benchmark: Simple straight road navigation."""
        g = WeightedGraph()
        # Straight road: 10 nodes in a line
        for i in range(10):
            g.add_node(i, float(i * 10), 0.0)
        for i in range(9):
            g.add_edge(i, i + 1)

        router = RouteRouter(g, activation_radius=15.0)
        # Start between nodes, goal near end
        result = router.plan(5.0, 2.0, 85.0, 2.0)

        assert result is not None
        assert result.start_node == 0 or result.start_node == 1
        assert result.goal_node == 8 or result.goal_node == 9
        # Path should be sequential
        for i in range(len(result.path) - 1):
            assert abs(result.path[i + 1] - result.path[i]) == 1

    def test_benchmark_intersection(self):
        """Benchmark: T-intersection navigation."""
        g = WeightedGraph()
        # T-junction:
        #     1
        #     |
        # 0 - 2 - 3
        g.add_node(0, 0.0, 0.0)
        g.add_node(1, 10.0, 10.0)
        g.add_node(2, 10.0, 0.0)
        g.add_node(3, 20.0, 0.0)
        g.add_edge(0, 2)
        g.add_edge(1, 2)
        g.add_edge(2, 3)

        router = RouteRouter(g, activation_radius=5.0)
        # Route from left to top
        result = router.plan(1.0, 0.0, 10.0, 9.0)

        assert result is not None
        assert result.start_node == 0
        assert result.goal_node == 1
        # Must pass through junction node 2
        assert 2 in result.path

    def test_benchmark_off_road_rejection(self):
        """Benchmark: Off-road position outside activation radius is rejected."""
        g = _grid_graph()
        router = RouteRouter(g, activation_radius=10.0)
        # Position far off-road
        result = router.plan(100.0, 100.0, 10.0, 10.0)
        assert result is None

    def test_benchmark_disconnected_network(self):
        """Benchmark: Disconnected road network fails gracefully."""
        g = WeightedGraph()
        # Two separate road segments
        g.add_node(0, 0.0, 0.0)
        g.add_node(1, 10.0, 0.0)
        g.add_edge(0, 1)

        g.add_node(2, 100.0, 0.0)
        g.add_node(3, 110.0, 0.0)
        g.add_edge(2, 3)

        router = RouteRouter(g, activation_radius=20.0)
        # Route from first segment to second segment
        result = router.plan(5.0, 0.0, 105.0, 0.0)
        assert result is None  # Should fail due to disconnection

    def test_benchmark_deterministic_projection(self):
        """Benchmark: Projection behavior is deterministic."""
        g = _grid_graph()
        router = RouteRouter(g, activation_radius=20.0)

        # Run same query multiple times
        results = []
        for _ in range(5):
            result = router.plan(12.0, 13.0, 18.0, 17.0)
            results.append(result)

        # All results should be identical
        assert all(r is not None for r in results)
        paths = [tuple(r.path) for r in results]
        assert len(set(paths)) == 1  # All paths identical
