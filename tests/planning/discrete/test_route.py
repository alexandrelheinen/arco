"""Unit tests for RouteRouter."""

from __future__ import annotations

import math

import numpy as np
import pytest

from arco.mapping.graph import CartesianGraph
from arco.planning.discrete import RouteRouter


def _grid_graph() -> CartesianGraph:
    """Return a simple 3x3 grid graph for testing.

    Layout:
        0 - 1 - 2
        |   |   |
        3 - 4 - 5
        |   |   |
        6 - 7 - 8

    Each node is 10 units apart.
    """
    g = CartesianGraph()
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
        result = router.plan(np.array([0.0, 0.0]), np.array([20.0, 20.0]))

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
        result = router.plan(np.array([0.5, 0.5]), np.array([19.5, 19.5]))

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
        result = router.plan(np.array([10.0, 10.0]), np.array([10.1, 10.1]))

        assert result is not None
        assert result.start_node == result.goal_node == 4
        assert result.path == [4]

    def test_plan_disconnected_graph(self):
        """Route fails on disconnected graph."""
        g = CartesianGraph()
        g.add_node(0, 0.0, 0.0)
        g.add_node(1, 100.0, 100.0)

        router = RouteRouter(g)
        result = router.plan(np.array([0.0, 0.0]), np.array([100.0, 100.0]))
        assert result is None

    def test_plan_path_length(self):
        """Route path length is reasonable."""
        g = _grid_graph()
        router = RouteRouter(g)
        result = router.plan(np.array([0.0, 0.0]), np.array([20.0, 0.0]))

        assert result is not None
        assert len(result.path) == 3
        assert result.path == [0, 1, 2]


class TestRouteRouterActivationRadius:
    """Test activation radius constraints."""

    def test_start_outside_radius_fails(self):
        """Route fails when start is outside activation radius."""
        g = _grid_graph()
        router = RouteRouter(g, activation_radius=5.0)
        result = router.plan(np.array([50.0, 50.0]), np.array([10.0, 10.0]))
        assert result is None

    def test_goal_outside_radius_fails(self):
        """Route fails when goal is outside activation radius."""
        g = _grid_graph()
        router = RouteRouter(g, activation_radius=5.0)
        result = router.plan(np.array([10.0, 10.0]), np.array([50.0, 50.0]))
        assert result is None

    def test_both_within_radius_succeeds(self):
        """Route succeeds when both start and goal are within radius."""
        g = _grid_graph()
        router = RouteRouter(g, activation_radius=5.0)
        result = router.plan(np.array([2.0, 2.0]), np.array([18.0, 18.0]))
        assert result is not None
        assert result.start_node == 0
        assert result.goal_node == 8

    def test_no_radius_allows_any_distance(self):
        """Route succeeds at any distance when radius is None."""
        g = _grid_graph()
        router = RouteRouter(g, activation_radius=None)
        result = router.plan(
            np.array([-100.0, -100.0]), np.array([100.0, 100.0])
        )
        assert result is not None
        assert result.start_node == 0
        assert result.goal_node == 8


class TestRouteRouterProjection:
    """Test projection metadata in results."""

    def test_projection_coordinates(self):
        """Result includes projection coordinates."""
        g = _grid_graph()
        router = RouteRouter(g)
        result = router.plan(np.array([5.0, 5.0]), np.array([15.0, 15.0]))

        assert result is not None
        start_node_pos = g.position(result.start_node)
        assert np.allclose(result.start_projection, start_node_pos)

        goal_node_pos = g.position(result.goal_node)
        assert np.allclose(result.goal_projection, goal_node_pos)

    def test_projection_distances(self):
        """Result includes projection distances."""
        g = _grid_graph()
        router = RouteRouter(g)
        result = router.plan(np.array([1.0, 2.0]), np.array([19.0, 18.0]))

        assert result is not None
        expected_start_dist = math.hypot(1.0, 2.0)
        assert math.isclose(
            result.start_distance, expected_start_dist, abs_tol=1e-6
        )

        expected_goal_dist = math.hypot(1.0, 2.0)
        assert math.isclose(
            result.goal_distance, expected_goal_dist, abs_tol=1e-6
        )


class TestRouteRouterEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_graph(self):
        """Route fails on empty graph."""
        g = CartesianGraph()
        router = RouteRouter(g)
        result = router.plan(np.array([0.0, 0.0]), np.array([10.0, 10.0]))
        assert result is None

    def test_single_node_graph(self):
        """Route succeeds on single-node graph if start and goal snap to it."""
        g = CartesianGraph()
        g.add_node(0, 5.0, 5.0)
        router = RouteRouter(g)
        result = router.plan(np.array([4.0, 4.0]), np.array([6.0, 6.0]))

        assert result is not None
        assert result.start_node == 0
        assert result.goal_node == 0
        assert result.path == [0]

    def test_optimal_path_selection(self):
        """Route selects the optimal (shortest) path."""
        g = CartesianGraph()
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
        result = router.plan(np.array([0.0, 0.0]), np.array([2.0, 0.0]))

        assert result is not None
        assert result.path == [0, 3]


class TestRouteRouterDocumentedBenchmarks:
    """Test documented benchmark scenarios for acceptance criteria."""

    def test_benchmark_straight_road(self):
        """Benchmark: Simple straight road navigation."""
        g = CartesianGraph()
        for i in range(10):
            g.add_node(i, float(i * 10), 0.0)
        for i in range(9):
            g.add_edge(i, i + 1)

        router = RouteRouter(g, activation_radius=15.0)
        result = router.plan(np.array([5.0, 2.0]), np.array([85.0, 2.0]))

        assert result is not None
        assert result.start_node == 0 or result.start_node == 1
        assert result.goal_node == 8 or result.goal_node == 9
        for i in range(len(result.path) - 1):
            assert abs(result.path[i + 1] - result.path[i]) == 1

    def test_benchmark_intersection(self):
        """Benchmark: T-intersection navigation."""
        g = CartesianGraph()
        g.add_node(0, 0.0, 0.0)
        g.add_node(1, 10.0, 10.0)
        g.add_node(2, 10.0, 0.0)
        g.add_node(3, 20.0, 0.0)
        g.add_edge(0, 2)
        g.add_edge(1, 2)
        g.add_edge(2, 3)

        router = RouteRouter(g, activation_radius=5.0)
        result = router.plan(np.array([1.0, 0.0]), np.array([10.0, 9.0]))

        assert result is not None
        assert result.start_node == 0
        assert result.goal_node == 1
        assert 2 in result.path

    def test_benchmark_off_road_rejection(self):
        """Benchmark: Off-road position outside activation radius is rejected."""
        g = _grid_graph()
        router = RouteRouter(g, activation_radius=10.0)
        result = router.plan(np.array([100.0, 100.0]), np.array([10.0, 10.0]))
        assert result is None

    def test_benchmark_disconnected_network(self):
        """Benchmark: Disconnected road network fails gracefully."""
        g = CartesianGraph()
        g.add_node(0, 0.0, 0.0)
        g.add_node(1, 10.0, 0.0)
        g.add_edge(0, 1)

        g.add_node(2, 100.0, 0.0)
        g.add_node(3, 110.0, 0.0)
        g.add_edge(2, 3)

        router = RouteRouter(g, activation_radius=20.0)
        result = router.plan(np.array([5.0, 0.0]), np.array([105.0, 0.0]))
        assert result is None

    def test_benchmark_deterministic_projection(self):
        """Benchmark: Projection behavior is deterministic."""
        g = _grid_graph()
        router = RouteRouter(g, activation_radius=20.0)

        results = []
        for _ in range(5):
            result = router.plan(
                np.array([12.0, 13.0]), np.array([18.0, 17.0])
            )
            results.append(result)

        assert all(r is not None for r in results)
        paths = [tuple(r.path) for r in results]
        assert len(set(paths)) == 1
