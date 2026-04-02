"""Test RouteRouter integration with RoadGraph."""

from arco.mapping.graph import RoadGraph
from arco.planning.discrete import RouteRouter


def test_route_router_with_road_graph():
    """Verify RouteRouter works with RoadGraph (inherits projection methods)."""
    # Create a simple road graph with waypoints
    graph = RoadGraph()
    graph.add_node(0, 0.0, 0.0)
    graph.add_node(1, 10.0, 0.0)
    graph.add_node(2, 20.0, 0.0)

    # Add edges with waypoints (for future spline interpolation)
    graph.add_edge(0, 1, waypoints=[(5.0, 1.0)])  # Curved road
    graph.add_edge(1, 2, waypoints=[(15.0, -1.0)])  # Another curved road

    # Create router and plan route
    router = RouteRouter(graph, activation_radius=5.0)
    result = router.plan(1.0, 1.0, 19.0, 1.0)

    assert result is not None
    assert result.path == [0, 1, 2]
    assert result.start_node == 0
    assert result.goal_node == 2


def test_road_graph_inherits_projection_methods():
    """Verify RoadGraph inherits WeightedGraph projection methods."""
    graph = RoadGraph()
    graph.add_node(0, 0.0, 0.0)
    graph.add_node(1, 10.0, 10.0)
    graph.add_edge(0, 1, waypoints=[(5.0, 5.0)])

    # Test find_nearest_node
    nearest = graph.find_nearest_node(1.0, 1.0)
    assert nearest == 0

    # Test heuristic
    assert graph.heuristic(0, 1) > 0

    # Test project_to_nearest_edge
    projection = graph.project_to_nearest_edge(5.0, 5.0)
    assert projection is not None
