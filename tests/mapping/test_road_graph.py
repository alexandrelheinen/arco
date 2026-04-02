"""Unit tests for RoadGraph."""

from __future__ import annotations

import math

import pytest

from arco.mapping.graph import RoadGraph


def _simple_road() -> RoadGraph:
    """Return a simple road: two intersections connected by a curved road."""
    g = RoadGraph()
    g.add_node(0, 0.0, 0.0)
    g.add_node(1, 100.0, 0.0)
    waypoints = [(25.0, 5.0), (50.0, -3.0), (75.0, 2.0)]
    g.add_edge(0, 1, waypoints=waypoints)
    return g


class TestRoadGraphConstruction:
    def test_add_nodes(self):
        g = RoadGraph()
        g.add_node(0, 1.0, 2.0)
        g.add_node(1, 4.0, 6.0)
        assert sorted(g.nodes) == [0, 1]

    def test_add_edge_without_waypoints(self):
        g = RoadGraph()
        g.add_node(0, 0.0, 0.0)
        g.add_node(1, 10.0, 0.0)
        g.add_edge(0, 1)
        assert g.edge_geometry(0, 1) == []

    def test_add_edge_with_waypoints(self):
        g = _simple_road()
        waypoints = g.edge_geometry(0, 1)
        assert len(waypoints) == 3
        assert waypoints[0] == (25.0, 5.0)
        assert waypoints[1] == (50.0, -3.0)
        assert waypoints[2] == (75.0, 2.0)

    def test_edge_geometry_symmetric(self):
        """Edge geometry should work in both directions."""
        g = _simple_road()
        assert g.edge_geometry(0, 1) == g.edge_geometry(1, 0)

    def test_edge_geometry_nonexistent_edge(self):
        """Querying geometry of non-existent edge returns empty list."""
        g = RoadGraph()
        g.add_node(0, 0.0, 0.0)
        g.add_node(1, 10.0, 0.0)
        # No edge added
        assert g.edge_geometry(0, 1) == []


class TestRoadGraphFullGeometry:
    def test_full_geometry_forward(self):
        """Full geometry should include start, waypoints, and end in order."""
        g = _simple_road()
        full_geom = g.full_edge_geometry(0, 1)
        assert len(full_geom) == 5  # start + 3 waypoints + end
        assert full_geom[0] == (0.0, 0.0)  # start node
        assert full_geom[1] == (25.0, 5.0)  # first waypoint
        assert full_geom[2] == (50.0, -3.0)  # second waypoint
        assert full_geom[3] == (75.0, 2.0)  # third waypoint
        assert full_geom[4] == (100.0, 0.0)  # end node

    def test_full_geometry_reverse(self):
        """Full geometry in reverse should reverse waypoint order."""
        g = _simple_road()
        full_geom = g.full_edge_geometry(1, 0)
        assert len(full_geom) == 5
        assert full_geom[0] == (100.0, 0.0)  # start (was end)
        assert full_geom[1] == (75.0, 2.0)  # reverse waypoints
        assert full_geom[2] == (50.0, -3.0)
        assert full_geom[3] == (25.0, 5.0)
        assert full_geom[4] == (0.0, 0.0)  # end (was start)

    def test_full_geometry_no_waypoints(self):
        """Full geometry with no waypoints is just start and end."""
        g = RoadGraph()
        g.add_node(0, 0.0, 0.0)
        g.add_node(1, 10.0, 5.0)
        g.add_edge(0, 1)  # no waypoints
        full_geom = g.full_edge_geometry(0, 1)
        assert len(full_geom) == 2
        assert full_geom[0] == (0.0, 0.0)
        assert full_geom[1] == (10.0, 5.0)


class TestRoadGraphWeightedGraphCompatibility:
    """Ensure RoadGraph maintains WeightedGraph functionality."""

    def test_distance_method(self):
        g = _simple_road()
        # Distance should be Euclidean between nodes
        assert math.isclose(g.distance(0, 1), 100.0)

    def test_neighbors(self):
        g = _simple_road()
        assert list(g.neighbors(0)) == [1]
        assert list(g.neighbors(1)) == [0]

    def test_astar_compatibility(self):
        """RoadGraph should work with AStarPlanner."""
        from arco.planning.discrete.astar import AStarPlanner

        g = RoadGraph()
        # Simple chain: 0 - 1 - 2
        for i in range(3):
            g.add_node(i, float(i * 10), 0.0)
        g.add_edge(0, 1, waypoints=[(5.0, 1.0)])
        g.add_edge(1, 2, waypoints=[(15.0, -1.0)])

        planner = AStarPlanner(g)
        path = planner.plan(0, 2)
        assert path is not None
        assert path == [0, 1, 2]


class TestRoadGraphMultipleEdges:
    def test_triangle_network(self):
        """Test a simple triangle road network."""
        g = RoadGraph()
        g.add_node(0, 0.0, 0.0)
        g.add_node(1, 100.0, 0.0)
        g.add_node(2, 50.0, 87.0)

        g.add_edge(0, 1, waypoints=[(50.0, -10.0)])
        g.add_edge(1, 2, waypoints=[(75.0, 30.0)])
        g.add_edge(2, 0, waypoints=[(25.0, 30.0)])

        # Check all edges have geometry
        assert len(g.edge_geometry(0, 1)) == 1
        assert len(g.edge_geometry(1, 2)) == 1
        assert len(g.edge_geometry(2, 0)) == 1

        # Check connectivity
        assert set(g.neighbors(0)) == {1, 2}
        assert set(g.neighbors(1)) == {0, 2}
        assert set(g.neighbors(2)) == {0, 1}
