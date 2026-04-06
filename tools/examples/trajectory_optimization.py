"""
Trajectory Optimization example using RRT* + TrajectoryOptimizer.

Demonstrates the two-stage trajectory optimization pipeline:

1. RRT* finds a collision-free reference path.
2. :class:`~arco.planning.continuous.TrajectoryOptimizer` refines the
   reference path into a time-optimal trajectory, penalising total
   traversal time, deviation from the reference, speed mismatch, and
   obstacle proximity.

Usage
-----
Run interactively (opens a matplotlib window)::

    python tools/examples/trajectory_optimization.py

Save the output image without opening a window::

    python tools/examples/trajectory_optimization.py --save path/to/output.png
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from logging_config import configure_logging
from viewer.occupancy import draw_occupancy

from arco.guidance.vehicle import DubinsVehicle
from arco.mapping import KDTreeOccupancy
from arco.planning.continuous import (
    RRTPlanner,
    TrajectoryOptimizer,
    TrajectoryResult,
)
from config import load_config

logger = logging.getLogger(__name__)

_rrt_cfg = load_config("rrt")
_opt_cfg = load_config("optimizer")


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
    rng = np.random.default_rng(42)
    x_max = float(_rrt_cfg["bounds"][0][1])
    y_max = float(_rrt_cfg["bounds"][1][1])

    # Central horizontal wall with a gap
    wall_pts = [[x, y_max / 2.0] for x in np.arange(0.0, x_max * 0.6, 1.5)] + [
        [x, y_max / 2.0] for x in np.arange(x_max * 0.7, x_max, 1.5)
    ]

    # Random scattered obstacles
    margin = 6.0
    scatter_count = 30
    scatter_pts: list[list[float]] = []
    while len(scatter_pts) < scatter_count:
        p = rng.uniform([margin, margin], [x_max - margin, y_max - margin])
        scatter_pts.append(p.tolist())

    all_pts = wall_pts + scatter_pts
    return KDTreeOccupancy(
        all_pts, clearance=float(_rrt_cfg["obstacle_clearance"])
    )


def main(save_path: str | None = None) -> None:
    """Run RRT* followed by trajectory optimization and visualize.

    Args:
        save_path: If provided, save the figure to this path instead of
            opening an interactive window.
    """
    if save_path is not None:
        matplotlib.use("Agg")

    bounds = [tuple(b) for b in _rrt_cfg["bounds"]]
    occ = build_occupancy()

    start = np.array([2.0, 2.0])
    goal = np.array(
        [
            float(_rrt_cfg["bounds"][0][1]) - 2.0,
            float(_rrt_cfg["bounds"][1][1]) - 2.0,
        ]
    )

    # ------------------------------------------------------------------ #
    # Stage 0 — RRT* global planner                                       #
    # ------------------------------------------------------------------ #
    planner = RRTPlanner(
        occ,
        bounds=bounds,
        max_sample_count=int(_rrt_cfg["max_sample_count"]),
        step_size=float(_rrt_cfg["step_size"]),
        goal_tolerance=float(_rrt_cfg["goal_tolerance"]),
        collision_check_count=int(_rrt_cfg["collision_check_count"]),
        goal_bias=float(_rrt_cfg["goal_bias"]),
        early_stop=bool(_rrt_cfg.get("early_stop", True)),
    )

    logger.info("Running RRT* …")
    planner_t0 = time.perf_counter()
    tree_nodes, _, reference_path = planner.get_tree(start, goal)
    planner_elapsed = time.perf_counter() - planner_t0
    planner_nodes = len(tree_nodes)
    planner_steps = (
        max(0, len(reference_path) - 1) if reference_path is not None else 0
    )
    planned_len = _polyline_length(reference_path)

    if reference_path is None:
        logger.error("RRT* failed to find a path — aborting example.")
        return

    logger.info("Reference path: %d waypoints", len(reference_path))

    # ------------------------------------------------------------------ #
    # Stage 1+2 — TrajectoryOptimizer                                     #
    # ------------------------------------------------------------------ #
    vehicle = DubinsVehicle(
        max_speed=10.0,
        min_speed=0.0,
        max_turn_rate=2.0,
    )

    optimizer = TrajectoryOptimizer(
        occ,
        cruise_speed=float(_opt_cfg["cruise_speed"]),
        weight_time=float(_opt_cfg["weight_time"]),
        weight_deviation=float(_opt_cfg["weight_deviation"]),
        weight_velocity=float(_opt_cfg["weight_velocity"]),
        weight_collision=float(_opt_cfg["weight_collision"]),
        time_relaxation=float(_opt_cfg["time_relaxation"]),
        method=str(_opt_cfg["method"]),
        sample_count=int(_opt_cfg["sample_count"]),
        max_iter=int(_opt_cfg.get("max_iter", 500)),
        ftol=float(_opt_cfg.get("ftol", 1e-9)),
    )

    logger.info("Running TrajectoryOptimizer …")
    opt_t0 = time.perf_counter()
    result: TrajectoryResult = optimizer.optimize(
        reference_path,
        inverse_kinematics=vehicle.inverse_kinematics,
        feasibility=vehicle.is_feasible,
    )
    _ = time.perf_counter() - opt_t0

    total_time = sum(result.durations)
    traj_arc_len = _polyline_length(result.states)
    path_status = "found" if result.is_feasible else "stalled"
    optimizer_status = (
        f"{result.optimizer_status_code}: {result.optimizer_status_text}"
    )
    logger.info(
        "Optimized trajectory: cost=%.3f, total_time=%.2fs",
        result.cost,
        total_time,
    )

    # ------------------------------------------------------------------ #
    # Visualization                                                        #
    # ------------------------------------------------------------------ #
    fig, ax = draw_occupancy(
        occ,
        bounds=bounds,
        path=reference_path,
        start=start,
        goal=goal,
        path_color="steelblue",
        title="Trajectory Optimization",
    )

    metrics_lines = [
        f"Planner steps / nodes: {planner_steps} / {planner_nodes}",
        f"Planner time: {_format_clock(planner_elapsed)}",
        f"Planned path length: {int(round(planned_len))} m",
        f"Trajectory arc length: {int(round(traj_arc_len))} m",
        f"Trajectory duration: {_format_clock(total_time)}",
        f"Path status: {path_status}",
        f"Optimizer status: {optimizer_status}",
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

    # Overlay optimized trajectory
    opt_pts = np.array(result.states)
    ax.plot(
        opt_pts[:, 0],
        opt_pts[:, 1],
        "o-",
        color="orangered",
        linewidth=2.5,
        markersize=5,
        zorder=5,
        label="Optimized",
    )

    # Annotate segment durations
    for i, (t_i, p) in enumerate(zip(result.durations, result.states)):
        ax.annotate(
            f"{t_i:.1f}s",
            xy=(p[0], p[1]),
            fontsize=6,
            color="orangered",
            ha="center",
            va="bottom",
            zorder=6,
        )

    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info("Saved trajectory optimization example to %s", save_path)
    else:
        plt.show()


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save",
        metavar="PATH",
        default=None,
        help=(
            "Save the figure to PATH instead of opening an interactive"
            " window."
        ),
    )
    args = parser.parse_args()
    main(save_path=args.save)
