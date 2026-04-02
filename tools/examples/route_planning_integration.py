#!/usr/bin/env python
"""Example: Route planning on procedurally generated road networks.

This example demonstrates the integration of:
1. RoadNetworkGenerator - procedural road graph generation with waypoints
2. RouteRouter - route planning with continuous coordinate projection
3. RoadGraph - weighted graph with per-edge geometry metadata

This shows the complete pipeline for Phase 1.2 of the horse auto-follow system.
"""

from arco.mapping.generator import RoadNetworkGenerator
from arco.planning.discrete import RouteRouter


def main():
    """Run route planning on a procedurally generated road network."""
    print("=" * 70)
    print("Route Planning on Procedurally Generated Road Network")
    print("=" * 70)

    # Create generator with fixed seed for reproducibility
    generator = RoadNetworkGenerator(seed=42)

    # Generate a 5×5 grid road network
    print("\nGenerating 5×5 grid road network...")
    graph = generator.generate_grid_network(
        grid_size=(5, 5), cell_size=50.0, waypoints_per_edge=3, curvature=0.3
    )

    print(f"  Nodes: {len(graph.nodes)}")
    print(f"  Edges: {len(graph.edges)}")

    # Check waypoints on a few edges
    edge_0_1 = graph.edge_geometry(0, 1)
    print(f"  Edge 0→1 has {len(edge_0_1)} waypoints")

    # Create router with activation radius
    activation_radius = 30.0
    router = RouteRouter(graph, activation_radius=activation_radius)
    print(f"\nRouter activation radius: {activation_radius} units")

    # Example 1: Route from top-left to bottom-right
    print("\n" + "-" * 70)
    print("Example 1: Diagonal route across the grid")
    print("-" * 70)
    start_x, start_y = 5.0, 5.0
    goal_x, goal_y = 195.0, 195.0

    print(f"Start position: ({start_x:.1f}, {start_y:.1f})")
    print(f"Goal position: ({goal_x:.1f}, {goal_y:.1f})")

    result = router.plan(start_x, start_y, goal_x, goal_y)
    if result is not None:
        print(f"\n✓ Route found!")
        print(f"  Path length: {len(result.path)} intersections")
        print(f"  Path: {result.path[:5]}...{result.path[-3:]}")
        print(f"  Start projection: {result.start_projection}")
        print(f"  Goal projection: {result.goal_projection}")

        # Show geometry for first edge in path
        if len(result.path) >= 2:
            edge_waypoints = graph.edge_geometry(
                result.path[0], result.path[1]
            )
            print(f"  First edge waypoints: {len(edge_waypoints)} points")
            if edge_waypoints:
                print(f"    First waypoint: {edge_waypoints[0]}")

    # Example 2: Route with curved roads
    print("\n" + "-" * 70)
    print("Example 2: Generate organic road network with curves")
    print("-" * 70)

    # Generate organic network
    generator_curved = RoadNetworkGenerator(seed=123)
    graph_curved = generator_curved.generate_random_network(
        num_intersections=15,
        area=200.0,
        connect_radius=80.0,
        waypoints_per_edge=5,
        curvature=0.8,
    )

    print(
        f"Organic network: {len(graph_curved.nodes)} nodes, {len(graph_curved.edges)} edges"
    )

    # Plan route on curved network
    router_curved = RouteRouter(graph_curved, activation_radius=40.0)
    result_curved = router_curved.plan(20.0, 20.0, 180.0, 180.0)

    if result_curved is not None:
        print(f"\n✓ Route found on organic network!")
        print(f"  Path length: {len(result_curved.path)} nodes")
        print(f"  Path: {result_curved.path}")

        # Show total geometry waypoints along route
        total_waypoints = 0
        for i in range(len(result_curved.path) - 1):
            waypoints = graph_curved.edge_geometry(
                result_curved.path[i], result_curved.path[i + 1]
            )
            total_waypoints += len(waypoints)
        print(f"  Total waypoints along route: {total_waypoints}")
        print(f"  (These can be used for spline interpolation in Phase 1.3)")
    else:
        print("\n✗ No route found (nodes may be too disconnected)")

    # Example 3: Show path smoothing preparation
    print("\n" + "-" * 70)
    print("Example 3: Prepare data for path smoothing (Phase 1.3)")
    print("-" * 70)

    if result is not None and len(result.path) >= 3:
        print(f"Original discrete path: {result.path[:5]}")

        # Collect full geometry for first few edges
        full_geometry = []
        for i in range(min(3, len(result.path) - 1)):
            edge_geom = graph.full_edge_geometry(
                result.path[i], result.path[i + 1]
            )
            full_geometry.extend(edge_geom[:-1])  # Avoid duplicating endpoints
        full_geometry.append(graph.position(result.path[2]))

        print(f"\nFull geometry for first 3 edges:")
        print(f"  Total points: {len(full_geometry)}")
        print(f"  First 3 points: {full_geometry[:3]}")
        print(f"\n  → These points will be fed to B-spline interpolation")
        print(
            f"  → Result: smooth continuous path for Pure Pursuit controller"
        )

    print("\n" + "=" * 70)
    print("Integration demonstration complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  - Phase 1.3: Path smoothing (B-spline/Catmull-Rom interpolation)")
    print("  - Phase 1.4: Pure Pursuit controller for path tracking")
    print("  - Phase 2.1: Dynamic replanning for moving targets")


if __name__ == "__main__":
    main()
