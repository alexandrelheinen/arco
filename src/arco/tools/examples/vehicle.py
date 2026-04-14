"""Vehicle benchmark (2-D) - RRT* vs SST in one matplotlib figure.

This scenario compares both planners on the same scattered-obstacle map.

Figure layout (3 subplots in a row)
-------------------------------------
* Left   — Combined workspace: both RRT* and SST paths/executed trajectories,
  obstacle heatmap.
* Middle — C-space (workspace = C-space for 2-D Dubins position): obstacle
  heatmap with velocity reach circle (blue) around start and goal.
* Right  — Lyapunov function V(t) = ‖(x,y)(t) − goal‖ for executed
  trajectories with a sliding-window highlight of the last T/10 seconds.

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
import matplotlib.pyplot as plt
import numpy as np

from arco.config.palette import annotation_hex, layer_hex, obstacle_hex
from arco.mapping import KDTreeOccupancy
from arco.planning.continuous import (
    RRTPlanner,
    SSTPlanner,
    TrajectoryOptimizer,
    TrajectoryPruner,
)
from arco.tools.logging_config import configure_logging
from arco.tools.simulator.sim.tracking import VehicleConfig, build_vehicle_sim
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
) -> tuple[list[np.ndarray] | None, float, str, list[float]]:
    if path is None or len(path) < 2:
        return None, 0.0, "no-path", []
    pruner = TrajectoryPruner(occ)
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
            (f"{res.optimizer_status_code}: {res.optimizer_status_text}"),
            durs,
        )
    except Exception:
        logger.exception("Trajectory optimization failed")
        return None, 0.0, "exception", []


def _simulate_vehicle(
    traj: list[np.ndarray] | None,
    occ: KDTreeOccupancy,
    vehicle_cfg: dict,
    dt: float = 0.05,
) -> list[tuple[float, float, float]]:
    """Run a headless TrackingLoop and return the executed (x, y, θ) poses.

    The full Dubins state is (x m, y m, θ rad): position from the planner
    waypoints, heading θ from the TrackingLoop at each step.

    Args:
        traj: Optimised trajectory states (each at least (x, y, …)).
        occ: Occupancy map for optional repulsion.
        vehicle_cfg: ``vehicle`` sub-dict from the scenario YAML.
        dt: Control time step in seconds.

    Returns:
        List of ``(x, y, θ)`` poses.  Empty when *traj* is ``None`` or short.
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
    # Full Dubins state: (x m, y m, θ rad) — position from planner, heading from tracking.
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
    rrt_nodes, _, rrt_path = rrt.get_tree(start.copy(), goal.copy())
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
    sst_nodes, _, sst_path = sst.get_tree(start.copy(), goal.copy())
    sst_time = time.perf_counter() - t0

    rrt_traj, rrt_dur, rrt_opt, rrt_durs = _optimize(
        occ, rrt_path, vehicle_cfg
    )
    sst_traj, sst_dur, sst_opt, sst_durs = _optimize(
        occ, sst_path, vehicle_cfg
    )

    logger.info("Simulating RRT* executed trajectory …")
    rrt_executed = _simulate_vehicle(rrt_traj or rrt_path, occ, vehicle_cfg)
    logger.info("Simulating SST executed trajectory …")
    sst_executed = _simulate_vehicle(sst_traj or sst_path, occ, vehicle_cfg)

    import matplotlib.patches as mpatches

    def _lyapunov_series(
        traj_states: list[tuple[float, float, float]] | None,
        goal: np.ndarray,
        dt: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute time axis and V(t) = ||(x,y)(t) − goal|| for a trajectory.

        Uses the simulation step ``dt`` to produce a time axis aligned with
        every simulation step in *traj_states*.

        Args:
            traj_states: Sequence of ``(x, y, θ)`` executed poses (one per
                simulation step).
            goal: 2-D goal position ``[gx, gy]``.
            dt: Simulation step size in seconds (must match the dt used in
                :func:`_simulate_vehicle`).

        Returns:
            Tuple of ``(times, V)`` arrays.  Both empty if input is too short.
        """
        if traj_states is None or len(traj_states) < 2:
            return np.array([]), np.array([])
        times = np.arange(len(traj_states), dtype=float) * dt
        V = np.array(
            [
                math.hypot(s[0] - float(goal[0]), s[1] - float(goal[1]))
                for s in traj_states
            ]
        )
        return times, V

    fig, (ax_ws, ax_cs, ax_lv) = plt.subplots(1, 3, figsize=(18, 6))

    # ---- ax_ws: Combined workspace with both planners' results -------------
    fig_tmp, ax_ws = draw_occupancy(
        occ,
        bounds=bounds,
        path=rrt_path,
        tree_nodes=None,
        tree_parent=None,
        start=start,
        goal=goal,
        draw_tree=False,
        path_alpha=(0.35 if rrt_traj is not None else 1.0),
        title="Vehicle benchmark — Cartesian workspace",
        ax=ax_ws,
    )
    _ = fig_tmp

    # Overlay SST path
    if sst_path is not None and len(sst_path) >= 2:
        sarr = np.array(sst_path)
        ax_ws.plot(
            sarr[:, 0],
            sarr[:, 1],
            color=layer_hex("sst", "path"),
            linewidth=1.5,
            alpha=(0.35 if sst_traj is not None else 0.9),
            label="SST path",
        )

    if rrt_traj is not None and len(rrt_traj) >= 2:
        arr = np.array(rrt_traj)
        ax_ws.plot(
            arr[:, 0],
            arr[:, 1],
            "o-",
            color=layer_hex("rrt", "trajectory"),
            linewidth=2.3,
            markersize=3,
            label="RRT* traj",
        )
    if len(rrt_executed) >= 2:
        ex = np.array(rrt_executed)
        ax_ws.plot(
            ex[:, 0],
            ex[:, 1],
            color=layer_hex("rrt", "vehicle"),
            linewidth=1.8,
            linestyle="--",
            alpha=0.85,
            label="RRT* executed",
        )
    if sst_traj is not None and len(sst_traj) >= 2:
        arr = np.array(sst_traj)
        ax_ws.plot(
            arr[:, 0],
            arr[:, 1],
            "o-",
            color=layer_hex("sst", "trajectory"),
            linewidth=2.3,
            markersize=3,
            label="SST traj",
        )
    if len(sst_executed) >= 2:
        ex = np.array(sst_executed)
        ax_ws.plot(
            ex[:, 0],
            ex[:, 1],
            color=layer_hex("sst", "vehicle"),
            linewidth=1.8,
            linestyle="--",
            alpha=0.85,
            label="SST executed",
        )

    ax_ws.text(
        0.02,
        0.98,
        "\n".join(
            [
                f"RRT* steps/nodes: {max(0, len(rrt_path)-1 if rrt_path else 0)}/{len(rrt_nodes)}",
                f"RRT* time: {rrt_time:.2f}s | len: {_polyline_length(rrt_path):.1f} m",
                f"RRT* traj dur: {rrt_dur:.1f}s | {rrt_opt}",
                f"SST steps/nodes: {max(0, len(sst_path)-1 if sst_path else 0)}/{len(sst_nodes)}",
                f"SST time: {sst_time:.2f}s | len: {_polyline_length(sst_path):.1f} m",
                f"SST traj dur: {sst_dur:.1f}s | {sst_opt}",
            ]
        ),
        transform=ax_ws.transAxes,
        va="top",
        ha="left",
        fontsize=7,
        color="black",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
    )
    ax_ws.grid(True, alpha=0.3)
    ax_ws.legend(loc="upper right", fontsize=7)

    # ---- ax_cs: C-space = workspace (2-D position) with vel. reach circles --
    fig_tmp2, ax_cs = draw_occupancy(
        occ,
        bounds=bounds,
        path=rrt_path,
        tree_nodes=None,
        tree_parent=None,
        start=start,
        goal=goal,
        draw_tree=False,
        path_alpha=0.4,
        title="C-space (x m, y m) — velocity constraints",
        ax=ax_cs,
    )
    _ = fig_tmp2

    if sst_path is not None and len(sst_path) >= 2:
        sarr = np.array(sst_path)
        ax_cs.plot(
            sarr[:, 0],
            sarr[:, 1],
            color=layer_hex("sst", "path"),
            linewidth=1.5,
            alpha=0.4,
        )

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

    # ---- ax_lv: Lyapunov V(t) = ‖(x,y)(t) − goal‖ -------------------------
    rrt_exec_times, rrt_V = _lyapunov_series(
        rrt_executed if rrt_executed else None, goal
    )
    sst_exec_times, sst_V = _lyapunov_series(
        sst_executed if sst_executed else None, goal
    )

    if len(rrt_exec_times) > 0:
        ax_lv.plot(
            rrt_exec_times,
            rrt_V,
            color=layer_hex("rrt", "trajectory"),
            linewidth=1.8,
            label="RRT* V(t)",
        )
        window = (rrt_exec_times[-1] - rrt_exec_times[0]) / 10.0
        ax_lv.axvspan(
            rrt_exec_times[-1] - window,
            rrt_exec_times[-1],
            alpha=0.10,
            color="gray",
        )
    if len(sst_exec_times) > 0:
        ax_lv.plot(
            sst_exec_times,
            sst_V,
            color=layer_hex("sst", "trajectory"),
            linewidth=1.8,
            label="SST V(t)",
        )
        window = (sst_exec_times[-1] - sst_exec_times[0]) / 10.0
        ax_lv.axvspan(
            sst_exec_times[-1] - window,
            sst_exec_times[-1],
            alpha=0.10,
            color="steelblue",
        )

    ax_lv.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax_lv.set_xlabel("Time (s)")
    ax_lv.set_ylabel("V(t) = ‖(x,y) − goal‖ (m)")
    ax_lv.set_title("Lyapunov function")
    ax_lv.legend(loc="upper right", fontsize=7)
    ax_lv.grid(True, alpha=0.3)

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
