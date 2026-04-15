"""Vehicle benchmark (2-D) — RRT* vs SST in one matplotlib figure.

This scenario compares both planners on the same scattered-obstacle map.

Figure layout (standard two-frame)
------------------------------------
* Top-left  — Workspace: obstacle heatmap + both RRT* and SST paths,
  optimised trajectories, and executed traces overlaid.
* Top-right — C-space (workspace = C-space for 2-D Dubins position):
  obstacle heatmap with velocity reach circles around start and goal.
* Bottom    — Metrics: per-planner step counts, planning times, path and
  trajectory lengths.

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
import math
import os
import time

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from arco.mapping import KDTreeOccupancy
from arco.planning.continuous import (
    RRTPlanner,
    SSTPlanner,
    TrajectoryOptimizer,
    TrajectoryPruner,
)
from arco.tools.logging_config import configure_logging
from arco.tools.simulator.sim.tracking import VehicleConfig, build_vehicle_sim
from arco.tools.viewer import FrameRenderer, SceneSnapshot
from arco.tools.viewer.layout import StandardLayout
from arco.tools.viewer.utils import parent_dict_to_list, polyline_length

logger = logging.getLogger(__name__)


def _build_vehicle_snapshot(
    planner: str,
    occ: KDTreeOccupancy,
    start: np.ndarray,
    goal: np.ndarray,
    nodes: list[np.ndarray],
    parent_dict: dict[int, int | None],
    path: list[np.ndarray] | None,
    traj: list[np.ndarray] | None,
    executed: list[tuple[float, float, float]],
    *,
    include_obstacles: bool = True,
) -> SceneSnapshot:
    """Build a SceneSnapshot from vehicle benchmark planning results.

    Args:
        planner: Planner key, e.g. ``"rrt"`` or ``"sst"``.
        occ: Occupancy map (obstacle point cloud).
        start: Start position array.
        goal: Goal position array.
        nodes: Exploration tree nodes.
        parent_dict: Dict mapping node index to parent index (None for root).
        path: Raw planned path.
        traj: Optimised trajectory states.
        executed: Executed ``(x, y, θ)`` poses from the tracking loop.
        include_obstacles: When ``False`` the obstacles list is left empty.

    Returns:
        A :class:`~arco.tools.viewer.SceneSnapshot` for the given planner.
    """
    obs: list[list[float]] = (
        [[float(v) for v in pt] for pt in occ.points]
        if include_obstacles
        else []
    )
    n = len(nodes)
    return SceneSnapshot.from_planning_result(
        scenario="vehicle",
        planner=planner,
        start=[float(start[0]), float(start[1])],
        goal=[float(goal[0]), float(goal[1])],
        obstacles=obs,
        tree_nodes=[[float(v) for v in nd] for nd in nodes] if nodes else [],
        tree_parent=parent_dict_to_list(parent_dict, n) if n > 0 else [],
        found_path=([[float(v) for v in p] for p in path] if path else None),
        adjusted_trajectory=(
            [[float(v) for v in p] for p in traj] if traj else None
        ),
        executed_trajectory=(
            [[float(x), float(y), float(th)] for x, y, th in executed]
            if len(executed) >= 2
            else None
        ),
    )


def build_occupancy(planner_cfg: dict, world_cfg: dict) -> KDTreeOccupancy:
    """Build the scattered-obstacle occupancy map.

    Args:
        planner_cfg: Planner sub-dict from the scenario YAML.
        world_cfg: World sub-dict from the scenario YAML.

    Returns:
        A :class:`~arco.mapping.KDTreeOccupancy` covering the planning bounds.
    """
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
    step_size: np.ndarray,
) -> tuple[list[np.ndarray] | None, float, str]:
    """Prune and trajectory-optimise a raw planned path.

    Args:
        occ: Occupancy map.
        path: Raw planned waypoints, or ``None``.
        vehicle_cfg: ``vehicle`` sub-dict from the scenario YAML.
        step_size: Planner step size array used by the pruner.

    Returns:
        Tuple of ``(traj, duration, status_string)``.
    """
    if path is None or len(path) < 2:
        return None, 0.0, "no-path"
    pruner = TrajectoryPruner(occ, step_size=step_size)
    path = pruner.prune(path)
    try:
        opt = TrajectoryOptimizer(
            occ,
            cruise_speed=float(vehicle_cfg["dubins"]["cruise_speed"]),
            weight_time=10.0,
            weight_deviation=1.0,
            weight_velocity=1.0,
            weight_collision=20.0,
            sample_count=10,
            max_iter=200,
        )
        res = opt.optimize(path)
        durs: list[float] = list(res.durations) if res.durations else []
        return (
            list(res.states),
            float(sum(durs)),
            f"{res.optimizer_status_code}: {res.optimizer_status_text}",
        )
    except Exception:
        logger.exception("Trajectory optimization failed")
        return None, 0.0, "exception"


def _simulate_vehicle(
    traj: list[np.ndarray] | None,
    occ: KDTreeOccupancy,
    vehicle_cfg: dict,
    dt: float = 0.05,
) -> list[tuple[float, float, float]]:
    """Run a headless TrackingLoop and return the executed (x, y, θ) poses.

    Args:
        traj: Optimised trajectory states (each at least (x, y, …)).
        occ: Occupancy map for optional repulsion.
        vehicle_cfg: ``vehicle`` sub-dict from the scenario YAML.
        dt: Control time step in seconds.

    Returns:
        List of ``(x, y, θ)`` poses.  Empty when *traj* is ``None`` or
        too short.
    """
    if traj is None or len(traj) < 2:
        return []
    dubins = vehicle_cfg.get("dubins", {})
    v_cfg = VehicleConfig(
        max_speed=float(dubins.get("max_speed", 5.0)),
        min_speed=0.0,
        cruise_speed=float(dubins.get("cruise_speed", 3.0)),
        lookahead_distance=float(dubins.get("lookahead", 4.0)),
        goal_radius=float(dubins.get("goal_radius", 3.0)),
        max_turn_rate=math.radians(float(dubins.get("max_turn_rate", 90.0))),
        max_acceleration=float(dubins.get("max_accel", 4.9)),
        max_turn_rate_dot=math.radians(
            float(dubins.get("max_turn_rate_dot", 3600.0))
        ),
        curvature_gain=float(dubins.get("curvature_gain", 0.5)),
    )
    waypoints: list[tuple[float, float]] = [
        (float(p[0]), float(p[1])) for p in traj
    ]
    _, loop = build_vehicle_sim(waypoints, v_cfg, occupancy=occ)
    executed: list[tuple[float, float, float]] = [
        (waypoints[0][0], waypoints[0][1], 0.0)
    ]
    max_steps = max(3000, len(waypoints) * 300)
    gx, gy = waypoints[-1]
    for _ in range(max_steps):
        result = loop.step(waypoints, dt)
        x, y, theta = result["pose"]
        executed.append((x, y, theta))
        if math.hypot(x - gx, y - gy) < v_cfg.goal_radius:
            break
    return executed


def main(cfg: dict, save_path: str | None = None) -> None:
    """Run the vehicle benchmark and display or save the figure.

    Args:
        cfg: Scenario configuration dictionary (loaded from
            ``vehicle.yml``).
        save_path: If provided, save the figure to this path instead of
            opening an interactive window.
    """
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
        step_size=planner_cfg["step_size"],
        goal_tolerance=float(planner_cfg["goal_tolerance"]),
        collision_check_count=int(planner_cfg["collision_check_count"]),
        goal_bias=float(planner_cfg["goal_bias"]),
        early_stop=bool(planner_cfg.get("early_stop", True)),
    )
    t0 = time.perf_counter()
    rrt_nodes, rrt_parent_dict, rrt_path = rrt.get_tree(
        start.copy(), goal.copy()
    )
    rrt_time = time.perf_counter() - t0

    sst = SSTPlanner(
        occ,
        bounds=bounds,
        max_sample_count=int(planner_cfg["sst_max_sample_count"]),
        step_size=planner_cfg["step_size"],
        goal_tolerance=float(planner_cfg["goal_tolerance"]),
        collision_check_count=int(planner_cfg["collision_check_count"]),
        goal_bias=float(planner_cfg["goal_bias"]),
        witness_radius=float(planner_cfg["witness_radius"]),
        early_stop=bool(planner_cfg.get("early_stop", True)),
    )
    t0 = time.perf_counter()
    sst_nodes, sst_parent_dict, sst_path = sst.get_tree(
        start.copy(), goal.copy()
    )
    sst_time = time.perf_counter() - t0

    _step_size = np.asarray(planner_cfg["step_size"], dtype=float)
    rrt_traj, rrt_dur, rrt_opt = _optimize(
        occ, rrt_path, vehicle_cfg, _step_size
    )
    sst_traj, sst_dur, sst_opt = _optimize(
        occ, sst_path, vehicle_cfg, _step_size
    )

    logger.info("Simulating RRT* executed trajectory …")
    rrt_executed = _simulate_vehicle(rrt_traj or rrt_path, occ, vehicle_cfg)
    logger.info("Simulating SST executed trajectory …")
    sst_executed = _simulate_vehicle(sst_traj or sst_path, occ, vehicle_cfg)

    rrt_snap = _build_vehicle_snapshot(
        "rrt",
        occ,
        start,
        goal,
        rrt_nodes,
        rrt_parent_dict,
        rrt_path,
        rrt_traj,
        rrt_executed,
        include_obstacles=True,
    )
    sst_snap = _build_vehicle_snapshot(
        "sst",
        occ,
        start,
        goal,
        sst_nodes,
        sst_parent_dict,
        sst_path,
        sst_traj,
        sst_executed,
        include_obstacles=False,
    )

    fig, ax_ws, ax_cs, ax_bottom = StandardLayout.create(
        title="Vehicle benchmark — RRT* vs SST"
    )

    # ---- ax_ws: Combined workspace -----------------------------------------
    FrameRenderer(draw_tree=False).render(ax_ws, rrt_snap)
    FrameRenderer(
        draw_tree=False, draw_obstacles=False, draw_start_goal=False
    ).render(ax_ws, sst_snap)
    ax_ws.grid(True, alpha=0.3)
    ax_ws.legend(loc="upper right", fontsize=7)

    # ---- ax_cs: C-space = workspace for 2-D Dubins -------------------------
    FrameRenderer(draw_tree=False, draw_obstacles=False).render(
        ax_cs, rrt_snap
    )
    FrameRenderer(
        draw_tree=False, draw_obstacles=False, draw_start_goal=False
    ).render(ax_cs, sst_snap)

    dubins_cfg = vehicle_cfg.get("dubins", {})
    cruise_speed = float(dubins_cfg.get("cruise_speed", 3.0))

    for center in (start, goal):
        circ = mpatches.Circle(
            (float(center[0]), float(center[1])),
            radius=cruise_speed,
            linewidth=1.2,
            edgecolor="blue",
            facecolor="none",
            alpha=0.6,
            linestyle="--",
        )
        ax_cs.add_patch(circ)
    ax_cs.grid(True, alpha=0.3)
    ax_cs.legend(loc="upper right", fontsize=7)

    # ---- Bottom: metrics ---------------------------------------------------
    StandardLayout.write_metrics(
        ax_bottom,
        [
            f"RRT*  steps/nodes: "
            f"{max(0, len(rrt_path)-1 if rrt_path else 0)}/{len(rrt_nodes)} | "
            f"time: {rrt_time:.2f} s | "
            f"path: {polyline_length(rrt_path):.1f} m | "
            f"traj: {rrt_dur:.1f} s | {rrt_opt}",
            f"SST   steps/nodes: "
            f"{max(0, len(sst_path)-1 if sst_path else 0)}/{len(sst_nodes)} | "
            f"time: {sst_time:.2f} s | "
            f"path: {polyline_length(sst_path):.1f} m | "
            f"traj: {sst_dur:.1f} s | {sst_opt}",
        ],
    )

    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
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
