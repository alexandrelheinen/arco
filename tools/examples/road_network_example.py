"""
A* pathfinding on a procedurally generated road network with spline-aware edges.

This example demonstrates:
1. Procedural generation of road networks (grid and random layouts)
2. Edge geometry metadata for curved roads
3. A* pathfinding on the generated network
4. Visualization of roads with their waypoints

Usage
-----
Run interactively (opens a matplotlib window)::

    python tools/examples/road_network_example.py

Save the output image without opening a window::

    python tools/examples/road_network_example.py --save path/to/output.png

Select network type::

    python tools/examples/road_network_example.py --type grid
    python tools/examples/road_network_example.py --type random
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Make the package importable when running the script directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
import matplotlib.pyplot as plt
from logging_config import configure_logging
from viewer.graph import draw_graph

from arco.mapping.generator import RoadNetworkGenerator
from arco.planning.discrete.astar import AStarPlanner

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
SEED = 42


def draw_road_network(graph, path=None, title="Road Network"):
    """Draw a road network with edge geometry waypoints.

    Args:
        graph: RoadGraph instance to visualize.
        path: Optional path sequence to highlight.
        title: Plot title.

    Returns:
        Tuple of (fig, ax) matplotlib objects.
    """
    fig, ax = draw_graph(graph, path, title=title)

    # Overlay the edge geometry waypoints
    for node_a, node_b, _ in graph.edges:
        waypoints = graph.edge_geometry(node_a, node_b)
        if waypoints:
            wx, wy = zip(*waypoints)
            ax.scatter(
                wx,
                wy,
                c="lightblue",
                s=20,
                alpha=0.6,
                zorder=2,
                label="Waypoints",
            )

            # Draw the full edge geometry as a polyline
            full_geom = graph.full_edge_geometry(node_a, node_b)
            fx, fy = zip(*full_geom)
            ax.plot(
                fx,
                fy,
                "c-",
                alpha=0.3,
                linewidth=1,
                zorder=1,
            )

    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    return fig, ax


def generate_grid_example(seed: int = SEED):
    """Generate and visualize a grid-based road network.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (graph, start, goal, path).
    """
    gen = RoadNetworkGenerator(seed=seed)
    graph = gen.generate_grid_network(
        grid_size=(4, 4),
        cell_size=100.0,
        waypoints_per_edge=4,
        curvature=0.2,
    )

    # Find path from top-left to bottom-right
    start = 0  # Top-left corner
    goal = 15  # Bottom-right corner (4*4 - 1)

    planner = AStarPlanner(graph)
    path = planner.plan(start, goal)

    return graph, start, goal, path


def generate_random_example(seed: int = SEED):
    """Generate and visualize a random road network.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (graph, start, goal, path).
    """
    gen = RoadNetworkGenerator(seed=seed)
    graph = gen.generate_random_network(
        num_intersections=25,
        area=400.0,
        connect_radius=120.0,
        waypoints_per_edge=3,
        curvature=0.15,
    )

    # Find path between two distant nodes
    nodes = graph.nodes
    if len(nodes) < 2:
        return graph, None, None, None

    # Simple heuristic: pick first and last node
    start = nodes[0]
    goal = nodes[-1]

    planner = AStarPlanner(graph)
    path = planner.plan(start, goal)

    return graph, start, goal, path


def main(network_type: str = "grid", save_path: str | None = None) -> None:
    """Generate and visualize a road network example.

    Args:
        network_type: Type of network to generate ("grid" or "random").
        save_path: Optional path to save the figure. If None, displays interactively.
    """
    if save_path is not None:
        matplotlib.use("Agg")

    if network_type == "grid":
        graph, start, goal, path = generate_grid_example()
        title = f"A* on 4×4 Grid Road Network (seed={SEED})"
    elif network_type == "random":
        graph, start, goal, path = generate_random_example()
        title = f"A* on Random Road Network (seed={SEED})"
    else:
        raise ValueError(f"Unknown network type: {network_type}")

    if path is not None:
        title += f"\nPath: {start} → {goal} ({len(path)} nodes)"
    else:
        title += f"\nNo path found: {start} → {goal}"

    fig, ax = draw_road_network(graph, path, title=title)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info("Saved road network example to %s", save_path)
    else:
        plt.show()


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--type",
        choices=["grid", "random"],
        default="grid",
        help="Type of road network to generate (default: grid)",
    )
    parser.add_argument(
        "--save",
        metavar="PATH",
        default=None,
        help="Save the figure to PATH instead of opening an interactive window.",
    )
    args = parser.parse_args()
    main(network_type=args.type, save_path=args.save)
