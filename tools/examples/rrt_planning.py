"""
RRT* planner on a 2-D continuous environment with scattered obstacles.

Demonstrates asymptotically optimal RRT* on a planar environment.  The
planner grows a tree from a start position and rewires edges to minimize
path cost.  An optional ``--tree`` flag renders the full exploration tree
for visual inspection.

Usage
-----
Run interactively (opens a matplotlib window)::

    python tools/examples/rrt_planning.py

Save the output image without opening a window::

    python tools/examples/rrt_planning.py --save path/to/output.png

Show the exploration tree as well::

    python tools/examples/rrt_planning.py --tree
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

# Make the package importable when running the script directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from logging_config import configure_logging
from viewer.occupancy import draw_occupancy

from arco.mapping import KDTreeOccupancy
from arco.planning.continuous import RRTPlanner, TrajectoryOptimizer
from config import load_config

logger = logging.getLogger(__name__)

_cfg = load_config("rrt")


def _format_clock(seconds: float) -> str:
    """Format seconds as ``MMminSSs`` rounded to whole seconds."""
    rounded = int(round(seconds))
    mins, secs = divmod(rounded, 60)
    return f"{mins:02d}min{secs:02d}s"


def _polyline_length(path: list[np.ndarray] | None) -> float:
    """Return total Euclidean arc length for a waypoint sequence."""
    if path is None or len(path) < 2:
        return 0.0
    return sum(
        float(np.linalg.norm(path[i + 1] - path[i]))
        for i in range(len(path) - 1)
    )


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
    """Run RRT* and visualize the result.

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
    planner_t0 = time.perf_counter()
    tree_nodes, tree_parent, path = planner.get_tree(start, goal)
    planner_elapsed = time.perf_counter() - planner_t0
    logger.info("Tree size: %d nodes", len(tree_nodes))

    planner_nodes = len(tree_nodes)
    planner_steps = max(0, (len(path) - 1) if path is not None else 0)
    path_found_text = "found" if path is not None else "stalled"
    path_len = _polyline_length(path)

    if path is not None:
        logger.info(
            "Path found: %d waypoints, length=%.2f", len(path), path_len
        )
    else:
        logger.warning("No path found.")

    # --- Trajectory optimization ----------------------------------------
    traj_states = None
    traj_duration = 0.0
    traj_arc_len = 0.0
    opt_status_text = "not-run"
    if path is not None:
        try:
            opt = TrajectoryOptimizer(
                occ,
                cruise_speed=3.0,
                weight_time=10.0,
                weight_deviation=1.0,
                weight_velocity=1.0,
                weight_collision=5.0,
                sample_count=2,
                max_iter=200,
            )
            opt_t0 = time.perf_counter()
            traj_result = opt.optimize(path)
            _ = time.perf_counter() - opt_t0
            traj_states = traj_result.states
            traj_duration = sum(traj_result.durations)
            traj_arc_len = _polyline_length(traj_result.states)
            opt_status_text = (
                f"{traj_result.optimizer_status_code}: "
                f"{traj_result.optimizer_status_text}"
            )
            logger.info(
                "Trajectory optimized: cost=%.3f, T=%.2fs",
                traj_result.cost,
                traj_duration,
            )
        except Exception:
            logger.exception("TrajectoryOptimizer failed; skipping overlay.")
            opt_status_text = "exception"

    tree_to_draw = (tree_nodes, tree_parent) if draw_tree else (None, None)
    # Draw raw path with reduced alpha so the trajectory stands out.
    path_alpha = 0.35 if traj_states is not None else 1.0
    fig, ax = draw_occupancy(
        occ,
        bounds=bounds,
        path=path,
        tree_nodes=tree_to_draw[0],
        tree_parent=tree_to_draw[1],
        start=start,
        goal=goal,
        draw_tree=draw_tree,
        path_alpha=path_alpha,
        title=f"RRT* — {int(_cfg['max_sample_count'])} samples",
    )

    metrics_lines = [
        f"Planner steps / nodes: {planner_steps} / {planner_nodes}",
        f"Planner time: {_format_clock(planner_elapsed)}",
        f"Planned path length: {int(round(path_len))} m",
        f"Trajectory arc length: {int(round(traj_arc_len))} m",
        f"Trajectory duration: {_format_clock(traj_duration)}",
        f"Path status: {path_found_text}",
        f"Optimizer status: {opt_status_text}",
    ]
    ax.text(
        0.01,
        0.99,
        "\n".join(metrics_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7,
        color="white",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "black",
            "alpha": 0.65,
            "edgecolor": "none",
        },
        zorder=10,
    )

    # Overlay the optimized trajectory in a highlighted color.
    if traj_states is not None and len(traj_states) >= 2:
        traj_arr = np.array([[p[0], p[1]] for p in traj_states])
        ax.plot(
            traj_arr[:, 0],
            traj_arr[:, 1],
            "o-",
            color="orangered",
            linewidth=2.5,
            markersize=3,
            zorder=6,
            label="Optimized trajectory",
        )
        ax.legend(loc="upper left", fontsize=8)

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
