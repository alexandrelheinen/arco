"""Unit tests for WeightedGraph."""

from __future__ import annotations

import math

import pytest

from arco.mapping.graph import WeightedGraph


def _triangle_graph() -> WeightedGraph:
    """Return a simple triangle graph: nodes 0, 1, 2 forming a right triangle."""
    g = WeightedGraph()
    g.add_node(0, 0.0, 0.0)
    g.add_node(1, 3.0, 0.0)
    g.add_node(2, 0.0, 4.0)
    g.add_edge(0, 1)  # weight defaults to distance = 3.0
    g.add_edge(0, 2)  # weight defaults to distance = 4.0
    g.add_edge(1, 2)  # weight defaults to distance = 5.0
    return g


class TestWeightedGraphConstruction:
    def test_add_nodes(self):
        g = WeightedGraph()
        g.add_node(0, 1.0, 2.0)
        g.add_node(1, 4.0, 6.0)
        assert sorted(g.nodes) == [0, 1]

    def test_position(self):
        g = WeightedGraph()
        g.add_node(7, 3.5, -1.0)
        assert g.position(7) == (3.5, -1.0)

    def test_add_edge_default_weight(self):
        g = _triangle_graph()
        assert math.isclose(g.distance(0, 1), 3.0)
        assert math.isclose(g.distance(0, 2), 4.0)
        assert math.isclose(g.distance(1, 2), 5.0)

    def test_add_edge_custom_weight(self):
        g = WeightedGraph()
        g.add_node(0, 0.0, 0.0)
        g.add_node(1, 10.0, 0.0)
        g.add_edge(0, 1, weight=1.0)
        assert math.isclose(g.distance(0, 1), 1.0)

    def test_edges_undirected(self):
        g = _triangle_graph()
        edges = g.edges
        # Three undirected edges
        assert len(edges) == 3
        pairs = {(min(a, b), max(a, b)) for a, b, _ in edges}
        assert (0, 1) in pairs
        assert (0, 2) in pairs
        assert (1, 2) in pairs

    def test_edges_no_duplicates(self):
        g = _triangle_graph()
        pairs = [(min(a, b), max(a, b)) for a, b, _ in g.edges]
        assert len(pairs) == len(set(pairs)), "Duplicate edges returned"


class TestWeightedGraphNeighbors:
    def test_neighbors(self):
        g = _triangle_graph()
        assert set(g.neighbors(0)) == {1, 2}
        assert set(g.neighbors(1)) == {0, 2}
        assert set(g.neighbors(2)) == {0, 1}

    def test_isolated_node_has_no_neighbors(self):
        g = WeightedGraph()
        g.add_node(0, 0.0, 0.0)
        assert list(g.neighbors(0)) == []

    def test_distance_fallback_euclidean(self):
        """distance() falls back to Euclidean when nodes are not connected."""
        g = WeightedGraph()
        g.add_node(0, 0.0, 0.0)
        g.add_node(1, 3.0, 4.0)
        # No edge added – should fall back to Euclidean distance 5.0
        assert math.isclose(g.distance(0, 1), 5.0)


class TestWeightedGraphAStarCompatibility:
    """Ensure WeightedGraph works end-to-end with AStarPlanner."""

    def test_astar_finds_path(self):
        from arco.planning.discrete.astar import AStarPlanner

        g = WeightedGraph()
        # Simple chain: 0 - 1 - 2 - 3
        for i in range(4):
            g.add_node(i, float(i), 0.0)
        for i in range(3):
            g.add_edge(i, i + 1)

        planner = AStarPlanner(g)
        path = planner.plan(0, 3)
        assert path is not None
        assert path[0] == 0
        assert path[-1] == 3

    def test_astar_no_path(self):
        from arco.planning.discrete.astar import AStarPlanner

        g = WeightedGraph()
        g.add_node(0, 0.0, 0.0)
        g.add_node(1, 1.0, 0.0)
        # No edges – no path possible
        planner = AStarPlanner(g)
        path = planner.plan(0, 1)
        assert path is None

    def test_astar_triangle_shortest(self):
        from arco.planning.discrete.astar import AStarPlanner

        g = _triangle_graph()
        planner = AStarPlanner(g)
        path = planner.plan(0, 2)
        assert path is not None
        # Direct edge 0→2 costs 4.0; going 0→1→2 costs 8.0, so shortest is direct.
        assert path == [0, 2]


class TestWeightedGraphProjection:
    """Test nearest-node and nearest-edge projection functionality."""

    def test_find_nearest_node_simple(self):
        g = _triangle_graph()
        # Query position close to node 0 at (0, 0)
        nearest = g.find_nearest_node(0.1, 0.1)
        assert nearest == 0

    def test_find_nearest_node_equidistant(self):
        g = WeightedGraph()
        g.add_node(0, 0.0, 0.0)
        g.add_node(1, 2.0, 0.0)
        # Midpoint – should return one of them consistently
        nearest = g.find_nearest_node(1.0, 0.0)
        assert nearest in {0, 1}

    def test_find_nearest_node_with_radius(self):
        g = _triangle_graph()
        # Query position at (10, 10) – far from all nodes
        nearest = g.find_nearest_node(10.0, 10.0, max_radius=1.0)
        assert nearest is None

    def test_find_nearest_node_within_radius(self):
        g = _triangle_graph()
        # Query position at (0.5, 0) – close to node 0 at (0, 0)
        nearest = g.find_nearest_node(0.5, 0.0, max_radius=1.0)
        assert nearest == 0

    def test_find_nearest_node_empty_graph(self):
        g = WeightedGraph()
        nearest = g.find_nearest_node(1.0, 2.0)
        assert nearest is None

    def test_project_to_nearest_edge_on_segment(self):
        g = _triangle_graph()
        # Project onto edge 0-1: from (0,0) to (3,0)
        # Query position at (1.5, 0.1) should project to (1.5, 0)
        result = g.project_to_nearest_edge(1.5, 0.1)
        assert result is not None
        proj, node_a, node_b, dist = result
        proj_x, proj_y = proj
        assert math.isclose(proj_x, 1.5, abs_tol=1e-6)
        assert math.isclose(proj_y, 0.0, abs_tol=1e-6)
        assert {node_a, node_b} == {0, 1}
        assert math.isclose(dist, 0.1, abs_tol=1e-6)

    def test_project_to_nearest_edge_at_endpoint(self):
        g = _triangle_graph()
        # Project onto edge from position beyond endpoint
        # Position (-1, 0) should clamp to node 0 at (0, 0)
        result = g.project_to_nearest_edge(-1.0, 0.0)
        assert result is not None
        proj, node_a, node_b, dist = result
        proj_x, proj_y = proj
        assert math.isclose(proj_x, 0.0, abs_tol=1e-6)
        assert math.isclose(proj_y, 0.0, abs_tol=1e-6)
        assert math.isclose(dist, 1.0, abs_tol=1e-6)

    def test_project_to_nearest_edge_perpendicular(self):
        g = WeightedGraph()
        g.add_node(0, 0.0, 0.0)
        g.add_node(1, 10.0, 0.0)
        g.add_edge(0, 1)
        # Query position at (5, 3) – perpendicular to midpoint of edge
        result = g.project_to_nearest_edge(5.0, 3.0)
        assert result is not None
        proj, node_a, node_b, dist = result
        proj_x, proj_y = proj
        assert math.isclose(proj_x, 5.0, abs_tol=1e-6)
        assert math.isclose(proj_y, 0.0, abs_tol=1e-6)
        assert math.isclose(dist, 3.0, abs_tol=1e-6)

    def test_project_to_nearest_edge_with_radius(self):
        g = _triangle_graph()
        # Query position far from graph
        result = g.project_to_nearest_edge(100.0, 100.0, max_radius=1.0)
        assert result is None

    def test_project_to_nearest_edge_empty_graph(self):
        g = WeightedGraph()
        result = g.project_to_nearest_edge(1.0, 2.0)
        assert result is None

    def test_heuristic_euclidean(self):
        """Test that heuristic method returns Euclidean distance."""
        g = _triangle_graph()
        # Distance from node 0 (0,0) to node 1 (3,0) is 3.0
        assert math.isclose(g.heuristic(0, 1), 3.0)
        # Distance from node 0 (0,0) to node 2 (0,4) is 4.0
        assert math.isclose(g.heuristic(0, 2), 4.0)
        # Distance from node 1 (3,0) to node 2 (0,4) is 5.0
        assert math.isclose(g.heuristic(1, 2), 5.0)
