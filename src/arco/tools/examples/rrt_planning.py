"""
RRT* planner on a 2-D continuous environment with scattered obstacles.

Demonstrates asymptotically optimal RRT* on a planar environment.  The
planner grows a tree from a start position and rewires edges to minimise
path cost.  An optional ``--tree`` flag renders the full exploration tree
for visual inspection.

Usage
-----
Run interactively (opens a matplotlib window)::

    python -m arco.tools.examples.rrt_planning

Save the output image without opening a window::

    python -m arco.tools.examples.rrt_planning --save path/to/output.png

Show the exploration tree as well::

    python -m arco.tools.examples.rrt_planning --tree
"""

from __future__ import annotations

import argparse
import logging
import os


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from arco.tools.logging_config import configure_logging
from arco.tools.viewer.occupancy import draw_occupancy

from arco.mapping import KDTreeOccupancy
from arco.planning.continuous import RRTPlanner
from arco.tools.config import load_config

logger = logging.getLogger(__name__)

_cfg = load_config("rrt")


def build_occupancy() -> KDTreeOccupancy:
    """Build a scattered obstacle environment for demonstration.

    Returns:
        A :class:`~arco.mapping.KDTreeOccupancy` with a central wall and
        scattered point obstacles.
    """
    rng = np.random.default_rng(7)
    x_max = float(_cfg["bounds"][0][1])
    y_max = float(_cfg["bounds"][1][1])

    # Central horizontal wall with a gap
    wall_pts = [[x, y_max / 2.0] for x in np.arange(0.0, x_max * 0.6, 1.5)] + [
        [x, y_max / 2.0] for x in np.arange(x_max * 0.7, x_max, 1.5)
    ]

    # Random scattered obstacles (avoid corners reserved for start/goal)
    margin = 5.0
    scatter_count = 40
    scatter_pts: list[list[float]] = []
    while len(scatter_pts) < scatter_count:
        p = rng.uniform([margin, margin], [x_max - margin, y_max - margin])
        scatter_pts.append(p.tolist())

    all_pts = wall_pts + scatter_pts
    return KDTreeOccupancy(
        all_pts, clearance=float(_cfg["obstacle_clearance"])
    )


def main(save_path: str | None = None, draw_tree: bool = False) -> None:
    """Run RRT* and visualise the result.

    Args:
        save_path: If provided, save the figure to this path instead of
            opening an interactive window.
        draw_tree: If ``True``, render the full exploration tree.
    """
    if save_path is not None:
        matplotlib.use("Agg")

    bounds = [tuple(b) for b in _cfg["bounds"]]
    occ = build_occupancy()

    start = np.array([2.0, 2.0])
    goal = np.array(
        [
            float(_cfg["bounds"][0][1]) - 2.0,
            float(_cfg["bounds"][1][1]) - 2.0,
        ]
    )

    planner = RRTPlanner(
        occ,
        bounds=bounds,
        max_sample_count=int(_cfg["max_sample_count"]),
        step_size=float(_cfg["step_size"]),
        goal_tolerance=float(_cfg["goal_tolerance"]),
        collision_check_count=int(_cfg["collision_check_count"]),
        goal_bias=float(_cfg["goal_bias"]),
        early_stop=bool(_cfg.get("early_stop", True)),
    )

    logger.info("Running RRT* …")
    tree_nodes, tree_parent, path = planner.get_tree(start, goal)
    logger.info("Tree size: %d nodes", len(tree_nodes))

    if path is not None:
        path_len = sum(
            float(np.linalg.norm(path[i + 1] - path[i]))
            for i in range(len(path) - 1)
        )
        logger.info(
            "Path found: %d waypoints, length=%.2f", len(path), path_len
        )
        subtitle = f"Path length: {path_len:.1f} | {len(path)} waypoints"
    else:
        logger.warning("No path found.")
        subtitle = "No path found"

    tree_to_draw = (tree_nodes, tree_parent) if draw_tree else (None, None)
    fig, ax = draw_occupancy(
        occ,
        bounds=bounds,
        path=path,
        tree_nodes=tree_to_draw[0],
        tree_parent=tree_to_draw[1],
        start=start,
        goal=goal,
        draw_tree=draw_tree,
        title=f"RRT* — {int(_cfg['max_sample_count'])} samples\n{subtitle}",
    )
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info("Saved RRT* example to %s", save_path)
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
    parser.add_argument(
        "--tree",
        action="store_true",
        default=False,
        help="Render the RRT* exploration tree.",
    )
    args = parser.parse_args()
    main(save_path=args.save, draw_tree=args.tree)
