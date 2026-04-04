"""
A* on a randomly generated weighted graph in a 2D plane.

Nodes and connections are generated automatically; the start and goal are
chosen as the two farthest-apart nodes in the graph.

Usage
-----
Run interactively (opens a matplotlib window)::

    python -m arco.tools.examples.astar_graph

Save the output image without opening a window::

    python -m arco.tools.examples.astar_graph --save path/to/output.png
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random

import matplotlib
import matplotlib.pyplot as plt

from arco.mapping.graph import CartesianGraph
from arco.planning.discrete.astar import AStarPlanner
from arco.tools.config import load_config
from arco.tools.logging_config import configure_logging
from arco.tools.viewer.graph import draw_graph

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameters (loaded from tools/config/)
# ---------------------------------------------------------------------------
_rng_cfg = load_config("random")
_graph_cfg = load_config("graph")


def build_random_graph(
    node_count: int = int(_graph_cfg["node_count"]),
    area: float = float(_graph_cfg["area"]),
    connect_radius: float = float(_graph_cfg["connect_radius"]),
    seed: int = int(_rng_cfg["seed"]),
) -> CartesianGraph:
    """Build a random planar graph with nodes connected by proximity.

    Args:
        node_count: Number of nodes to generate.
        area: Side length of the square placement area.
        connect_radius: Maximum distance for connecting two nodes.
        seed: Random seed for reproducibility.

    Returns:
        A :class:`~arco.mapping.graph.CartesianGraph` with at least one
        connected component containing the two extreme nodes.
    """
    rng = random.Random(seed)
    graph = CartesianGraph()

    for i in range(node_count):
        x = rng.uniform(0.0, area)
        y = rng.uniform(0.0, area)
        graph.add_node(i, x, y)

    # Connect nearby nodes.
    for i in range(node_count):
        for j in range(i + 1, node_count):
            xi, yi = graph.position(i)
            xj, yj = graph.position(j)
            if math.hypot(xi - xj, yi - yj) <= connect_radius:
                graph.add_edge(i, j)

    return graph


def find_farthest_pair(graph: CartesianGraph):
    """Return the pair of node IDs that are farthest apart (Euclidean)."""
    nodes = graph.nodes
    best_dist = -1.0
    best_pair = (nodes[0], nodes[-1])
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            xi, yi = graph.position(nodes[i])
            xj, yj = graph.position(nodes[j])
            d = math.hypot(xi - xj, yi - yj)
            if d > best_dist:
                best_dist = d
                best_pair = (nodes[i], nodes[j])
    return best_pair


def main(save_path: str | None = None) -> None:
    if save_path is not None:
        matplotlib.use("Agg")

    graph = build_random_graph()
    start, goal = find_farthest_pair(graph)

    planner = AStarPlanner(graph)
    path = planner.plan(start, goal)

    title = (
        f"A* on random graph — {int(_graph_cfg['node_count'])} nodes, r={float(_graph_cfg['connect_radius']):.0f}\n"
        f"Start: {start}  Goal: {goal}  "
        + (f"Path length: {len(path)}" if path else "No path found")
    )

    fig, ax = draw_graph(graph, path, title=title)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info("Saved graph example to %s", save_path)
    else:
        plt.show()


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save",
        metavar="PATH",
        default=None,
        help="Save the figure to PATH instead of opening an interactive window.",
    )
    args = parser.parse_args()
    main(save_path=args.save)
