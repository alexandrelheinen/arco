"""Test RouteRouter integration with RoadGraph."""

import numpy as np

from arco.mapping.graph import RoadGraph
from arco.planning.discrete import RouteRouter


def test_route_router_with_road_graph():
    """Verify RouteRouter works with RoadGraph (inherits projection methods)."""
    graph = RoadGraph()
    graph.add_node(0, 0.0, 0.0)
    graph.add_node(1, 10.0, 0.0)
    graph.add_node(2, 20.0, 0.0)

    graph.add_edge(0, 1, waypoints=[(5.0, 1.0)])
    graph.add_edge(1, 2, waypoints=[(15.0, -1.0)])

    router = RouteRouter(graph, activation_radius=5.0)
    result = router.plan(np.array([1.0, 1.0]), np.array([19.0, 1.0]))

    assert result is not None
    assert result.path == [0, 1, 2]
    assert result.start_node == 0
    assert result.goal_node == 2


def test_road_graph_inherits_projection_methods():
    """Verify RoadGraph inherits CartesianGraph projection methods."""
    graph = RoadGraph()
    graph.add_node(0, 0.0, 0.0)
    graph.add_node(1, 10.0, 10.0)
    graph.add_edge(0, 1, waypoints=[(5.0, 5.0)])

    nearest = graph.find_nearest_node(np.array([1.0, 1.0]))
    assert nearest == 0

    assert graph.heuristic(0, 1) > 0

    projection = graph.project_to_nearest_edge(np.array([5.0, 5.0]))
    assert projection is not None
