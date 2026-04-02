"""
A* on a randomly generated weighted graph in a 2D plane.

Nodes and connections are generated automatically; the start and goal are
chosen as the two farthest-apart nodes in the graph.

Usage
-----
Run interactively (opens a matplotlib window)::

    python tools/astar_graph_example.py

Save the output image without opening a window::

    python tools/astar_graph_example.py --save path/to/output.png
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import sys

# Make the package importable when running the script directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..")
)  # expose tools/viewer

import matplotlib
import matplotlib.pyplot as plt
from logging_config import configure_logging
from viewer.graph import draw_graph

from arco.mapping.graph import WeightedGraph
from arco.planning.discrete.astar import AStarPlanner

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
SEED = 42
NUM_NODES = 30
AREA = 100.0  # nodes are placed in [0, AREA] x [0, AREA]
CONNECT_RADIUS = 28.0  # nodes closer than this radius are connected


def build_random_graph(
    num_nodes: int = NUM_NODES,
    area: float = AREA,
    connect_radius: float = CONNECT_RADIUS,
    seed: int = SEED,
) -> WeightedGraph:
    """Build a random planar graph with nodes connected by proximity.

    Args:
        num_nodes: Number of nodes to generate.
        area: Side length of the square placement area.
        connect_radius: Maximum distance for connecting two nodes.
        seed: Random seed for reproducibility.

    Returns:
        A :class:`~arco.mapping.graph.WeightedGraph` with at least one
        connected component containing the two extreme nodes.
    """
    rng = random.Random(seed)
    graph = WeightedGraph()

    for i in range(num_nodes):
        x = rng.uniform(0.0, area)
        y = rng.uniform(0.0, area)
        graph.add_node(i, x, y)

    # Connect nearby nodes.
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            xi, yi = graph.position(i)
            xj, yj = graph.position(j)
            if math.hypot(xi - xj, yi - yj) <= connect_radius:
                graph.add_edge(i, j)

    return graph


def find_farthest_pair(graph: WeightedGraph):
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
        f"A* on random graph — {NUM_NODES} nodes, r={CONNECT_RADIUS:.0f}\n"
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
