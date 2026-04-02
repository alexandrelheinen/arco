#!/usr/bin/env python
"""Example: Route planning with continuous coordinates on a weighted graph.

This example demonstrates the RouteRouter planner for autonomous navigation
on road networks. It shows:
1. Creating a weighted graph representing a road network
2. Projecting continuous positions onto the graph
3. Planning routes with activation radius constraints
4. Handling failure modes (off-road, disconnected)

This is part of the horse auto-follow system (Phase 1.2).
"""

from arco.mapping.graph import WeightedGraph
from arco.planning.discrete import RouteRouter


def create_road_network() -> WeightedGraph:
    """Create a sample road network resembling a small town.

    Layout:
        0 - 1 - 2
        |   |   |
        3 - 4 - 5
        |   |   |
        6 - 7 - 8

    Each intersection is 50 units apart.
    """
    graph = WeightedGraph()

    # Add intersection nodes
    for i in range(3):
        for j in range(3):
            node_id = i * 3 + j
            x = float(j * 50)
            y = float(i * 50)
            graph.add_node(node_id, x, y)

    # Add horizontal roads
    for i in range(3):
        for j in range(2):
            node_a = i * 3 + j
            node_b = i * 3 + j + 1
            graph.add_edge(node_a, node_b)

    # Add vertical roads
    for i in range(2):
        for j in range(3):
            node_a = i * 3 + j
            node_b = (i + 1) * 3 + j
            graph.add_edge(node_a, node_b)

    return graph


def main():
    """Run route planning examples."""
    print("=" * 70)
    print("Route Planning Example: Continuous Coordinate Projection")
    print("=" * 70)

    # Create road network
    graph = create_road_network()
    print(
        f"\nRoad network: {len(graph.nodes)} intersections, {len(graph.edges)} roads"
    )

    # Create router with activation radius
    activation_radius = 30.0
    router = RouteRouter(graph, activation_radius=activation_radius)
    print(f"Activation radius: {activation_radius} units")

    # Example 1: Route from top-left to bottom-right
    print("\n" + "-" * 70)
    print("Example 1: Route from top-left to bottom-right")
    print("-" * 70)
    start_x, start_y = 5.0, 5.0
    goal_x, goal_y = 95.0, 95.0

    print(f"Start position: ({start_x}, {start_y})")
    print(f"Goal position: ({goal_x}, {goal_y})")

    result = router.plan(start_x, start_y, goal_x, goal_y)
    if result is not None:
        print(f"\n✓ Route found!")
        print(
            f"  Start node: {result.start_node} at {result.start_projection}"
        )
        print(f"  Goal node: {result.goal_node} at {result.goal_projection}")
        print(f"  Start distance: {result.start_distance:.2f} units")
        print(f"  Goal distance: {result.goal_distance:.2f} units")
        print(f"  Path length: {len(result.path)} intersections")
        print(f"  Path: {result.path}")
    else:
        print("✗ No route found")

    # Example 2: Route from center to right side
    print("\n" + "-" * 70)
    print("Example 2: Route from center to right side")
    print("-" * 70)
    start_x, start_y = 48.0, 52.0
    goal_x, goal_y = 98.0, 48.0

    print(f"Start position: ({start_x}, {start_y})")
    print(f"Goal position: ({goal_x}, {goal_y})")

    result = router.plan(start_x, start_y, goal_x, goal_y)
    if result is not None:
        print(f"\n✓ Route found!")
        print(
            f"  Start node: {result.start_node} at {result.start_projection}"
        )
        print(f"  Goal node: {result.goal_node} at {result.goal_projection}")
        print(f"  Path: {result.path}")
    else:
        print("✗ No route found")

    # Example 3: Off-road position (should fail)
    print("\n" + "-" * 70)
    print("Example 3: Off-road position (outside activation radius)")
    print("-" * 70)
    start_x, start_y = 200.0, 200.0  # Far from any road
    goal_x, goal_y = 50.0, 50.0

    print(f"Start position: ({start_x}, {start_y}) - OFF ROAD")
    print(f"Goal position: ({goal_x}, {goal_y})")

    result = router.plan(start_x, start_y, goal_x, goal_y)
    if result is not None:
        print(f"\n✓ Route found!")
        print(f"  Path: {result.path}")
    else:
        print("\n✗ No route found (start position outside activation radius)")

    # Example 4: Nearest node query
    print("\n" + "-" * 70)
    print("Example 4: Find nearest intersection to any position")
    print("-" * 70)
    query_x, query_y = 35.0, 75.0
    print(f"Query position: ({query_x}, {query_y})")

    nearest = graph.find_nearest_node(query_x, query_y)
    if nearest is not None:
        pos = graph.position(nearest)
        import math

        dist = math.hypot(query_x - pos[0], query_y - pos[1])
        print(f"Nearest intersection: {nearest} at {pos}")
        print(f"Distance: {dist:.2f} units")

    # Example 5: Project onto nearest edge
    print("\n" + "-" * 70)
    print("Example 5: Project position onto nearest road segment")
    print("-" * 70)
    query_x, query_y = 55.0, 25.0
    print(f"Query position: ({query_x}, {query_y})")

    projection = graph.project_to_nearest_edge(query_x, query_y)
    if projection is not None:
        proj_pos, node_a, node_b, dist = projection
        print(f"Projected point: {proj_pos}")
        print(f"Edge endpoints: {node_a} - {node_b}")
        print(f"Distance to road: {dist:.2f} units")

    print("\n" + "=" * 70)
    print("Route planning demonstration complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
