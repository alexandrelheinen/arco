#!/usr/bin/env python
"""Example: Route planning on procedurally generated road networks.

This example demonstrates the integration of:
1. RoadNetworkGenerator - procedural road graph generation with waypoints
2. RouteRouter - route planning with continuous coordinate projection
3. RoadGraph - weighted graph with per-edge geometry metadata

This shows the complete pipeline for Phase 1.2 of the horse auto-follow system.
"""

import logging
import os
import sys

# Make the package importable when running the script directly (without install).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
# Expose the tools/viewer and tools/config packages.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from logging_config import configure_logging

from arco.mapping.generator import RoadNetworkGenerator
from arco.planning.discrete import RouteRouter

logger = logging.getLogger(__name__)


def main():
    """Run route planning on a procedurally generated road network."""
    logger.info("=" * 70)
    logger.info("Route Planning on Procedurally Generated Road Network")
    logger.info("=" * 70)

    # Create generator with fixed seed for reproducibility
    generator = RoadNetworkGenerator(seed=42)

    # Generate a 5×5 grid road network
    logger.info("Generating 5\u00d75 grid road network...")
    graph = generator.generate_grid_network(
        grid_size=(5, 5),
        cell_size=50.0,
        waypoints_per_edge_count=3,
        curvature=0.3,
    )

    logger.info("  Nodes: %d", len(graph.nodes))
    logger.info("  Edges: %d", len(graph.edges))

    # Check waypoints on a few edges
    edge_0_1 = graph.edge_geometry(0, 1)
    logger.info("  Edge 0\u21921 has %d waypoints", len(edge_0_1))

    # Create router with activation radius
    activation_radius = 30.0
    router = RouteRouter(graph, activation_radius=activation_radius)
    logger.info("Router activation radius: %s units", activation_radius)

    # Example 1: Route from top-left to bottom-right
    logger.info("\n" + "-" * 70)
    logger.info("Example 1: Diagonal route across the grid")
    logger.info("-" * 70)
    start_x, start_y = 5.0, 5.0
    goal_x, goal_y = 195.0, 195.0

    logger.info("Start position: (%.1f, %.1f)", start_x, start_y)
    logger.info("Goal position: (%.1f, %.1f)", goal_x, goal_y)

    result = router.plan(
        np.array([start_x, start_y]), np.array([goal_x, goal_y])
    )
    if result is not None:
        logger.info("\u2713 Route found!")
        logger.info("  Path length: %d intersections", len(result.path))
        logger.info("  Path: %s...%s", result.path[:5], result.path[-3:])
        logger.info("  Start projection: %s", result.start_projection)
        logger.info("  Goal projection: %s", result.goal_projection)

        # Show geometry for first edge in path
        if len(result.path) >= 2:
            edge_waypoints = graph.edge_geometry(
                result.path[0], result.path[1]
            )
            logger.info(
                "  First edge waypoints: %d points", len(edge_waypoints)
            )
            if edge_waypoints:
                logger.info("    First waypoint: %s", edge_waypoints[0])

    # Example 2: Route with curved roads
    logger.info("\n" + "-" * 70)
    logger.info("Example 2: Generate organic road network with curves")
    logger.info("-" * 70)

    # Generate organic network
    generator_curved = RoadNetworkGenerator(seed=123)
    graph_curved = generator_curved.generate_random_network(
        intersection_count=15,
        area=200.0,
        connect_radius=80.0,
        waypoints_per_edge_count=5,
        curvature=0.8,
    )

    logger.info(
        "Organic network: %d nodes, %d edges",
        len(graph_curved.nodes),
        len(graph_curved.edges),
    )

    # Plan route on curved network
    router_curved = RouteRouter(graph_curved, activation_radius=40.0)
    result_curved = router_curved.plan(
        np.array([20.0, 20.0]), np.array([180.0, 180.0])
    )

    if result_curved is not None:
        logger.info("\u2713 Route found on organic network!")
        logger.info("  Path length: %d nodes", len(result_curved.path))
        logger.info("  Path: %s", result_curved.path)

        # Show total geometry waypoints along route
        total_waypoints = 0
        for i in range(len(result_curved.path) - 1):
            waypoints = graph_curved.edge_geometry(
                result_curved.path[i], result_curved.path[i + 1]
            )
            total_waypoints += len(waypoints)
        logger.info("  Total waypoints along route: %d", total_waypoints)
        logger.info(
            "  (These can be used for spline interpolation in Phase 1.3)"
        )
    else:
        logger.warning("\u2717 No route found (nodes may be too disconnected)")

    # Example 3: Show path smoothing preparation
    logger.info("\n" + "-" * 70)
    logger.info("Example 3: Prepare data for path smoothing (Phase 1.3)")
    logger.info("-" * 70)

    if result is not None and len(result.path) >= 3:
        logger.info("Original discrete path: %s", result.path[:5])

        # Collect full geometry for first few edges
        full_geometry = []
        for i in range(min(3, len(result.path) - 1)):
            edge_geom = graph.full_edge_geometry(
                result.path[i], result.path[i + 1]
            )
            full_geometry.extend(edge_geom[:-1])  # Avoid duplicating endpoints
        full_geometry.append(graph.position(result.path[2]))

        logger.info("Full geometry for first 3 edges:")
        logger.info("  Total points: %d", len(full_geometry))
        logger.info("  First 3 points: %s", full_geometry[:3])
        logger.info(
            "  \u2192 These points will be fed to B-spline interpolation"
        )
        logger.info(
            "  \u2192 Result: smooth continuous path for Pure Pursuit controller"
        )

    logger.info("\n" + "=" * 70)
    logger.info("Integration demonstration complete!")
    logger.info("=" * 70)
    logger.info("Next steps:")
    logger.info(
        "  - Phase 1.3: Path smoothing (B-spline/Catmull-Rom interpolation)"
    )
    logger.info("  - Phase 1.4: Pure Pursuit controller for path tracking")
    logger.info("  - Phase 2.1: Dynamic replanning for moving targets")


if __name__ == "__main__":
    configure_logging()
    main()
