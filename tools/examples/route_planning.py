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

import logging
import math

from arco.mapping.graph import WeightedGraph
from arco.planning.discrete import RouteRouter

logger = logging.getLogger(__name__)


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
    logger.info("=" * 70)
    logger.info("Route Planning Example: Continuous Coordinate Projection")
    logger.info("=" * 70)

    # Create road network
    graph = create_road_network()
    logger.info(
        "\nRoad network: %d intersections, %d roads",
        len(graph.nodes),
        len(graph.edges),
    )

    # Create router with activation radius
    activation_radius = 30.0
    router = RouteRouter(graph, activation_radius=activation_radius)
    logger.info("Activation radius: %s units", activation_radius)

    # Example 1: Route from top-left to bottom-right
    logger.info("\n" + "-" * 70)
    logger.info("Example 1: Route from top-left to bottom-right")
    logger.info("-" * 70)
    start_x, start_y = 5.0, 5.0
    goal_x, goal_y = 95.0, 95.0

    logger.info("Start position: (%s, %s)", start_x, start_y)
    logger.info("Goal position: (%s, %s)", goal_x, goal_y)

    result = router.plan(start_x, start_y, goal_x, goal_y)
    if result is not None:
        logger.info("\u2713 Route found!")
        logger.info(
            "  Start node: %s at %s",
            result.start_node,
            result.start_projection,
        )
        logger.info(
            "  Goal node: %s at %s", result.goal_node, result.goal_projection
        )
        logger.info("  Start distance: %.2f units", result.start_distance)
        logger.info("  Goal distance: %.2f units", result.goal_distance)
        logger.info("  Path length: %d intersections", len(result.path))
        logger.info("  Path: %s", result.path)
    else:
        logger.warning("\u2717 No route found")

    # Example 2: Route from center to right side
    logger.info("\n" + "-" * 70)
    logger.info("Example 2: Route from center to right side")
    logger.info("-" * 70)
    start_x, start_y = 48.0, 52.0
    goal_x, goal_y = 98.0, 48.0

    logger.info("Start position: (%s, %s)", start_x, start_y)
    logger.info("Goal position: (%s, %s)", goal_x, goal_y)

    result = router.plan(start_x, start_y, goal_x, goal_y)
    if result is not None:
        logger.info("\u2713 Route found!")
        logger.info(
            "  Start node: %s at %s",
            result.start_node,
            result.start_projection,
        )
        logger.info(
            "  Goal node: %s at %s", result.goal_node, result.goal_projection
        )
        logger.info("  Path: %s", result.path)
    else:
        logger.warning("\u2717 No route found")

    # Example 3: Off-road position (should fail)
    logger.info("\n" + "-" * 70)
    logger.info("Example 3: Off-road position (outside activation radius)")
    logger.info("-" * 70)
    start_x, start_y = 200.0, 200.0  # Far from any road
    goal_x, goal_y = 50.0, 50.0

    logger.info("Start position: (%s, %s) - OFF ROAD", start_x, start_y)
    logger.info("Goal position: (%s, %s)", goal_x, goal_y)

    result = router.plan(start_x, start_y, goal_x, goal_y)
    if result is not None:
        logger.info("\u2713 Route found!")
        logger.info("  Path: %s", result.path)
    else:
        logger.warning(
            "\u2717 No route found (start position outside activation radius)"
        )

    # Example 4: Nearest node query
    logger.info("\n" + "-" * 70)
    logger.info("Example 4: Find nearest intersection to any position")
    logger.info("-" * 70)
    query_x, query_y = 35.0, 75.0
    logger.info("Query position: (%s, %s)", query_x, query_y)

    nearest = graph.find_nearest_node(query_x, query_y)
    if nearest is not None:
        pos = graph.position(nearest)
        dist = math.hypot(query_x - pos[0], query_y - pos[1])
        logger.info("Nearest intersection: %s at %s", nearest, pos)
        logger.info("Distance: %.2f units", dist)

    # Example 5: Project onto nearest edge
    logger.info("\n" + "-" * 70)
    logger.info("Example 5: Project position onto nearest road segment")
    logger.info("-" * 70)
    query_x, query_y = 55.0, 25.0
    logger.info("Query position: (%s, %s)", query_x, query_y)

    projection = graph.project_to_nearest_edge(query_x, query_y)
    if projection is not None:
        proj_pos, node_a, node_b, dist = projection
        logger.info("Projected point: %s", proj_pos)
        logger.info("Edge endpoints: %s - %s", node_a, node_b)
        logger.info("Distance to road: %.2f units", dist)

    logger.info("\n" + "=" * 70)
    logger.info("Route planning demonstration complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from logging_config import configure_logging

    configure_logging()
    main()
