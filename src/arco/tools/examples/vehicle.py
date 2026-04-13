"""Vehicle benchmark (2-D) - RRT* vs SST in one matplotlib figure.

This scenario compares both planners on the same scattered-obstacle map and
shows planned paths plus optimized trajectories.

Usage
-----
Run interactively::

    python tools/examples/vehicle.py

Save image::

    python tools/examples/vehicle.py --save path/to/output.png
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from arco.mapping import KDTreeOccupancy
from arco.planning.continuous import (
    RRTPlanner,
    SSTPlanner,
    TrajectoryOptimizer,
)
from arco.tools.logging_config import configure_logging
from arco.tools.viewer.occupancy import draw_occupancy

logger = logging.getLogger(__name__)


def _polyline_length(path: list[np.ndarray] | None) -> float:
    if path is None or len(path) < 2:
        return 0.0
    return sum(
        float(np.linalg.norm(path[i + 1] - path[i]))
        for i in range(len(path) - 1)
    )


def build_occupancy(planner_cfg: dict, world_cfg: dict) -> KDTreeOccupancy:
    rng = np.random.default_rng(int(world_cfg.get("random_seed", 7)))
    x_max = float(planner_cfg["bounds"][0][1])
    y_max = float(planner_cfg["bounds"][1][1])

    wall_spacing = float(world_cfg.get("wall_spacing", 1.5))
    gap_a = float(world_cfg.get("wall_gap_start_fraction", 0.60))
    gap_b = float(world_cfg.get("wall_gap_end_fraction", 0.70))
    wall_pts = [
        [x, y_max / 2.0] for x in np.arange(0.0, x_max * gap_a, wall_spacing)
    ] + [
        [x, y_max / 2.0] for x in np.arange(x_max * gap_b, x_max, wall_spacing)
    ]

    margin = float(world_cfg.get("corner_margin", 5.0))
    scatter_count = int(world_cfg.get("scatter_count", 40))
    scatter_pts: list[list[float]] = []
    while len(scatter_pts) < scatter_count:
        p = rng.uniform([margin, margin], [x_max - margin, y_max - margin])
        scatter_pts.append(p.tolist())

    return KDTreeOccupancy(
        wall_pts + scatter_pts,
        clearance=float(planner_cfg["obstacle_clearance"]),
    )


def _optimize(
    occ: KDTreeOccupancy,
    path: list[np.ndarray] | None,
    vehicle_cfg: dict,
) -> tuple[list[np.ndarray] | None, float, str]:
    if path is None or len(path) < 2:
        return None, 0.0, "no-path"
    try:
        opt = TrajectoryOptimizer(
            occ,
            cruise_speed=float(vehicle_cfg["dubins"]["cruise_speed"]),
            weight_time=10.0,
            weight_deviation=1.0,
            weight_velocity=1.0,
            weight_collision=5.0,
            sample_count=2,
            max_iter=200,
        )
        res = opt.optimize(path)
        return (
            list(res.states),
            float(sum(res.durations)),
            (f"{res.optimizer_status_code}: {res.optimizer_status_text}"),
        )
    except Exception:
        logger.exception("Trajectory optimization failed")
        return None, 0.0, "exception"


def main(cfg: dict, save_path: str | None = None) -> None:
    if save_path is not None:
        matplotlib.use("Agg")

    planner_cfg = cfg.get("planner", cfg)
    world_cfg = cfg.get("world", {})
    vehicle_cfg = cfg.get("vehicle", {})

    occ = build_occupancy(planner_cfg, world_cfg)
    bounds = [tuple(b) for b in planner_cfg["bounds"]]
    start = np.array([2.0, 2.0])
    goal = np.array(
        [
            float(planner_cfg["bounds"][0][1]) - 2.0,
            float(planner_cfg["bounds"][1][1]) - 2.0,
        ]
    )

    rrt = RRTPlanner(
        occ,
        bounds=bounds,
        max_sample_count=int(planner_cfg["rrt_max_sample_count"]),
        step_size=float(planner_cfg["step_size"]),
        goal_tolerance=float(planner_cfg["goal_tolerance"]),
        collision_check_count=int(planner_cfg["collision_check_count"]),
        goal_bias=float(planner_cfg["goal_bias"]),
        early_stop=bool(planner_cfg.get("early_stop", True)),
    )
    t0 = time.perf_counter()
    rrt_nodes, _, rrt_path = rrt.get_tree(start.copy(), goal.copy())
    rrt_time = time.perf_counter() - t0

    sst = SSTPlanner(
        occ,
        bounds=bounds,
        max_sample_count=int(planner_cfg["sst_max_sample_count"]),
        step_size=float(planner_cfg["step_size"]),
        goal_tolerance=float(planner_cfg["goal_tolerance"]),
        collision_check_count=int(planner_cfg["collision_check_count"]),
        goal_bias=float(planner_cfg["goal_bias"]),
        witness_radius=float(planner_cfg["witness_radius"]),
        early_stop=bool(planner_cfg.get("early_stop", True)),
    )
    t0 = time.perf_counter()
    sst_nodes, _, sst_path = sst.get_tree(start.copy(), goal.copy())
    sst_time = time.perf_counter() - t0

    rrt_traj, rrt_dur, rrt_opt = _optimize(occ, rrt_path, vehicle_cfg)
    sst_traj, sst_dur, sst_opt = _optimize(occ, sst_path, vehicle_cfg)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))
    fig1, ax1 = draw_occupancy(
        occ,
        bounds=bounds,
        path=rrt_path,
        tree_nodes=None,
        tree_parent=None,
        start=start,
        goal=goal,
        draw_tree=False,
        path_alpha=(0.35 if rrt_traj is not None else 1.0),
        title="Vehicle benchmark - RRT*",
        ax=ax1,
    )
    _ = fig1
    if rrt_traj is not None and len(rrt_traj) >= 2:
        arr = np.array(rrt_traj)
        ax1.plot(
            arr[:, 0],
            arr[:, 1],
            "o-",
            color="orangered",
            linewidth=2.3,
            markersize=3,
            label="Optimized",
        )
    ax1.text(
        0.02,
        0.98,
        "\n".join(
            [
                f"Steps/nodes: {max(0, (len(rrt_path) - 1) if rrt_path else 0)}/{len(rrt_nodes)}",
                f"Planner: {rrt_time:.2f}s",
                f"Path length: {_polyline_length(rrt_path):.1f} m",
                f"Traj duration: {rrt_dur:.1f}s",
                f"Optimizer: {rrt_opt}",
            ]
        ),
        transform=ax1.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="black",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
    )
    ax1.grid(True, alpha=0.3)

    fig2, ax2 = draw_occupancy(
        occ,
        bounds=bounds,
        path=sst_path,
        tree_nodes=None,
        tree_parent=None,
        start=start,
        goal=goal,
        draw_tree=False,
        path_alpha=(0.35 if sst_traj is not None else 1.0),
        title="Vehicle benchmark - SST",
        ax=ax2,
    )
    _ = fig2
    if sst_traj is not None and len(sst_traj) >= 2:
        arr = np.array(sst_traj)
        ax2.plot(
            arr[:, 0],
            arr[:, 1],
            "o-",
            color="orangered",
            linewidth=2.3,
            markersize=3,
            label="Optimized",
        )
    ax2.text(
        0.02,
        0.98,
        "\n".join(
            [
                f"Steps/nodes: {max(0, (len(sst_path) - 1) if sst_path else 0)}/{len(sst_nodes)}",
                f"Planner: {sst_time:.2f}s",
                f"Path length: {_polyline_length(sst_path):.1f} m",
                f"Traj duration: {sst_dur:.1f}s",
                f"Optimizer: {sst_opt}",
            ]
        ),
        transform=ax2.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="black",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
    )
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info("Saved vehicle benchmark example to %s", save_path)
    else:
        plt.show()


if __name__ == "__main__":
    import yaml as _yaml

    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scenario", metavar="FILE", help="Path to scenario YAML file."
    )
    parser.add_argument("--save", metavar="PATH", default=None)
    args = parser.parse_args()
    with open(args.scenario) as _fh:
        _cfg = _yaml.safe_load(_fh) or {}
    main(_cfg, save_path=args.save)
