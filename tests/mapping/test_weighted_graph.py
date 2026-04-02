"""Unit tests for WeightedGraph and CartesianGraph."""

from __future__ import annotations

import math

import numpy as np
import pytest

from arco.mapping.graph import CartesianGraph, WeightedGraph


def _triangle_graph() -> CartesianGraph:
    """Return a simple triangle graph: nodes 0, 1, 2 forming a right triangle."""
    g = CartesianGraph()
    g.add_node(0, 0.0, 0.0)
    g.add_node(1, 3.0, 0.0)
    g.add_node(2, 0.0, 4.0)
    g.add_edge(0, 1)  # weight defaults to distance = 3.0
    g.add_edge(0, 2)  # weight defaults to distance = 4.0
    g.add_edge(1, 2)  # weight defaults to distance = 5.0
    return g


# ---------------------------------------------------------------------------
# WeightedGraph (generic, no spatial)
# ---------------------------------------------------------------------------


class TestWeightedGraphGeneric:
    def test_add_node(self):
        g = WeightedGraph()
        g.add_node(0)
        g.add_node(1)
        assert sorted(g.nodes) == [0, 1]

    def test_add_edge_requires_weight(self):
        g = WeightedGraph()
        g.add_node(0)
        g.add_node(1)
        g.add_edge(0, 1, weight=3.0)
        assert math.isclose(g.distance(0, 1), 3.0)

    def test_neighbors(self):
        g = WeightedGraph()
        g.add_node(0)
        g.add_node(1)
        g.add_node(2)
        g.add_edge(0, 1, weight=1.0)
        g.add_edge(0, 2, weight=2.0)
        assert set(g.neighbors(0)) == {1, 2}

    def test_edges_undirected(self):
        g = WeightedGraph()
        g.add_node(0)
        g.add_node(1)
        g.add_node(2)
        g.add_edge(0, 1, weight=1.0)
        g.add_edge(1, 2, weight=2.0)
        edges = g.edges
        assert len(edges) == 2
        pairs = {(min(a, b), max(a, b)) for a, b, _ in edges}
        assert (0, 1) in pairs
        assert (1, 2) in pairs

    def test_distance_no_edge_raises(self):
        g = WeightedGraph()
        g.add_node(0)
        g.add_node(1)
        with pytest.raises(KeyError):
            g.distance(0, 1)


# ---------------------------------------------------------------------------
# CartesianGraph construction
# ---------------------------------------------------------------------------


class TestCartesianGraphConstruction:
    def test_add_nodes(self):
        g = CartesianGraph()
        g.add_node(0, 1.0, 2.0)
        g.add_node(1, 4.0, 6.0)
        assert sorted(g.nodes) == [0, 1]

    def test_position_2d(self):
        g = CartesianGraph()
        g.add_node(7, 3.5, -1.0)
        assert np.allclose(g.position(7), [3.5, -1.0])

    def test_position_3d(self):
        g = CartesianGraph()
        g.add_node(0, 1.0, 2.0, 3.0)
        assert np.allclose(g.position(0), [1.0, 2.0, 3.0])

    def test_ndim_inferred(self):
        g = CartesianGraph()
        g.add_node(0, 1.0, 2.0)
        assert g.ndim == 2

    def test_ndim_specified(self):
        g = CartesianGraph(ndim=3)
        g.add_node(0, 1.0, 2.0, 3.0)
        assert g.ndim == 3

    def test_ndim_mismatch_raises(self):
        g = CartesianGraph()
        g.add_node(0, 1.0, 2.0)
        with pytest.raises(ValueError, match="dimension"):
            g.add_node(1, 1.0, 2.0, 3.0)

    def test_add_node_no_coords_raises(self):
        g = CartesianGraph()
        with pytest.raises(ValueError):
            g.add_node(0)

    def test_add_edge_default_weight(self):
        g = _triangle_graph()
        assert math.isclose(g.distance(0, 1), 3.0)
        assert math.isclose(g.distance(0, 2), 4.0)
        assert math.isclose(g.distance(1, 2), 5.0)

    def test_add_edge_custom_weight(self):
        g = CartesianGraph()
        g.add_node(0, 0.0, 0.0)
        g.add_node(1, 10.0, 0.0)
        g.add_edge(0, 1, weight=1.0)
        assert math.isclose(g.distance(0, 1), 1.0)

    def test_edges_undirected(self):
        g = _triangle_graph()
        edges = g.edges
        assert len(edges) == 3
        pairs = {(min(a, b), max(a, b)) for a, b, _ in edges}
        assert (0, 1) in pairs
        assert (0, 2) in pairs
        assert (1, 2) in pairs

    def test_edges_no_duplicates(self):
        g = _triangle_graph()
        pairs = [(min(a, b), max(a, b)) for a, b, _ in g.edges]
        assert len(pairs) == len(set(pairs)), "Duplicate edges returned"


# ---------------------------------------------------------------------------
# CartesianGraph neighbors
# ---------------------------------------------------------------------------


class TestCartesianGraphNeighbors:
    def test_neighbors(self):
        g = _triangle_graph()
        assert set(g.neighbors(0)) == {1, 2}
        assert set(g.neighbors(1)) == {0, 2}
        assert set(g.neighbors(2)) == {0, 1}

    def test_isolated_node_has_no_neighbors(self):
        g = CartesianGraph()
        g.add_node(0, 0.0, 0.0)
        assert list(g.neighbors(0)) == []

    def test_distance_fallback_euclidean(self):
        """distance() falls back to Euclidean when nodes are not connected."""
        g = CartesianGraph()
        g.add_node(0, 0.0, 0.0)
        g.add_node(1, 3.0, 4.0)
        # No edge added - should fall back to Euclidean distance 5.0
        assert math.isclose(g.distance(0, 1), 5.0)


# ---------------------------------------------------------------------------
# CartesianGraph A* compatibility
# ---------------------------------------------------------------------------


class TestCartesianGraphAStarCompatibility:
    """Ensure CartesianGraph works end-to-end with AStarPlanner."""

    def test_astar_finds_path(self):
        from arco.planning.discrete.astar import AStarPlanner

        g = CartesianGraph()
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

        g = CartesianGraph()
        g.add_node(0, 0.0, 0.0)
        g.add_node(1, 1.0, 0.0)
        planner = AStarPlanner(g)
        path = planner.plan(0, 1)
        assert path is None

    def test_astar_triangle_shortest(self):
        from arco.planning.discrete.astar import AStarPlanner

        g = _triangle_graph()
        planner = AStarPlanner(g)
        path = planner.plan(0, 2)
        assert path is not None
        assert path == [0, 2]


# ---------------------------------------------------------------------------
# CartesianGraph projection
# ---------------------------------------------------------------------------


class TestCartesianGraphProjection:
    """Test nearest-node and nearest-edge projection functionality."""

    def test_find_nearest_node_simple(self):
        g = _triangle_graph()
        nearest = g.find_nearest_node(np.array([0.1, 0.1]))
        assert nearest == 0

    def test_find_nearest_node_equidistant(self):
        g = CartesianGraph()
        g.add_node(0, 0.0, 0.0)
        g.add_node(1, 2.0, 0.0)
        nearest = g.find_nearest_node(np.array([1.0, 0.0]))
        assert nearest in {0, 1}

    def test_find_nearest_node_with_radius(self):
        g = _triangle_graph()
        nearest = g.find_nearest_node(np.array([10.0, 10.0]), max_radius=1.0)
        assert nearest is None

    def test_find_nearest_node_within_radius(self):
        g = _triangle_graph()
        nearest = g.find_nearest_node(np.array([0.5, 0.0]), max_radius=1.0)
        assert nearest == 0

    def test_find_nearest_node_empty_graph(self):
        g = CartesianGraph()
        nearest = g.find_nearest_node(np.array([1.0, 2.0]))
        assert nearest is None

    def test_project_to_nearest_edge_on_segment(self):
        g = _triangle_graph()
        result = g.project_to_nearest_edge(np.array([1.5, 0.1]))
        assert result is not None
        proj, node_a, node_b, dist = result
        assert math.isclose(proj[0], 1.5, abs_tol=1e-6)
        assert math.isclose(proj[1], 0.0, abs_tol=1e-6)
        assert {node_a, node_b} == {0, 1}
        assert math.isclose(dist, 0.1, abs_tol=1e-6)

    def test_project_to_nearest_edge_at_endpoint(self):
        g = _triangle_graph()
        result = g.project_to_nearest_edge(np.array([-1.0, 0.0]))
        assert result is not None
        proj, node_a, node_b, dist = result
        assert math.isclose(proj[0], 0.0, abs_tol=1e-6)
        assert math.isclose(proj[1], 0.0, abs_tol=1e-6)
        assert math.isclose(dist, 1.0, abs_tol=1e-6)

    def test_project_to_nearest_edge_perpendicular(self):
        g = CartesianGraph()
        g.add_node(0, 0.0, 0.0)
        g.add_node(1, 10.0, 0.0)
        g.add_edge(0, 1)
        result = g.project_to_nearest_edge(np.array([5.0, 3.0]))
        assert result is not None
        proj, node_a, node_b, dist = result
        assert math.isclose(proj[0], 5.0, abs_tol=1e-6)
        assert math.isclose(proj[1], 0.0, abs_tol=1e-6)
        assert math.isclose(dist, 3.0, abs_tol=1e-6)

    def test_project_to_nearest_edge_with_radius(self):
        g = _triangle_graph()
        result = g.project_to_nearest_edge(
            np.array([100.0, 100.0]), max_radius=1.0
        )
        assert result is None

    def test_project_to_nearest_edge_empty_graph(self):
        g = CartesianGraph()
        result = g.project_to_nearest_edge(np.array([1.0, 2.0]))
        assert result is None

    def test_heuristic_euclidean(self):
        """Test that heuristic method returns Euclidean distance."""
        g = _triangle_graph()
        assert math.isclose(g.heuristic(0, 1), 3.0)
        assert math.isclose(g.heuristic(0, 2), 4.0)
        assert math.isclose(g.heuristic(1, 2), 5.0)


# ---------------------------------------------------------------------------
# CartesianGraph 3D
# ---------------------------------------------------------------------------


class TestCartesianGraph3D:
    """Test CartesianGraph in 3 dimensions."""

    def _cube_graph(self) -> CartesianGraph:
        """Unit-length edges from the origin along each axis."""
        g = CartesianGraph()
        g.add_node(0, 0.0, 0.0, 0.0)
        g.add_node(1, 1.0, 0.0, 0.0)
        g.add_node(2, 0.0, 1.0, 0.0)
        g.add_node(3, 0.0, 0.0, 1.0)
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(0, 3)
        return g

    def test_3d_positions_stored(self):
        g = self._cube_graph()
        assert np.allclose(g.position(0), [0.0, 0.0, 0.0])
        assert np.allclose(g.position(1), [1.0, 0.0, 0.0])
        assert np.allclose(g.position(3), [0.0, 0.0, 1.0])

    def test_3d_edge_weight_euclidean(self):
        g = self._cube_graph()
        assert math.isclose(g.distance(0, 1), 1.0)
        assert math.isclose(g.distance(0, 2), 1.0)
        assert math.isclose(g.distance(0, 3), 1.0)

    def test_3d_heuristic(self):
        g = self._cube_graph()
        assert math.isclose(g.heuristic(1, 2), math.sqrt(2), abs_tol=1e-9)

    def test_3d_find_nearest_node(self):
        g = self._cube_graph()
        nearest = g.find_nearest_node(np.array([0.1, 0.1, 0.1]))
        assert nearest == 0

    def test_3d_find_nearest_node_with_radius(self):
        g = self._cube_graph()
        nearest = g.find_nearest_node(
            np.array([10.0, 10.0, 10.0]), max_radius=1.0
        )
        assert nearest is None

    def test_3d_ndim(self):
        g = self._cube_graph()
        assert g.ndim == 3

    def test_3d_astar(self):
        from arco.planning.discrete.astar import AStarPlanner

        g = self._cube_graph()
        # Add edge so there's a path: 1->0->2
        planner = AStarPlanner(g)
        path = planner.plan(1, 2)
        assert path is not None
        assert path[0] == 1
        assert path[-1] == 2
