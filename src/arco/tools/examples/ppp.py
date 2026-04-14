"""PPP (triple prismatic) robot planning in a 3-D warehouse environment.

Runs RRT* and SST planners side-by-side on an industrial warehouse bay
modelled as a box-shaped work volume (60 m × 20 m × 6 m) with ground-level
box obstacles. Three width-crossing barriers increase difficulty: a tall
barrier, a smaller one, then a split-half barrier with mixed heights.

For PPP robots the C-space is the workspace (all joints are prismatic), so
the C-space panel annotates the same 3-D volume with velocity constraints.

Figure layout (3 subplots in a row)
-------------------------------------
* Left   — Combined 3-D workspace: both RRT* and SST paths/trajectories.
* Middle — C-space (x m, y m, z m): obstacle boxes (red), both paths, velocity
  constraint box (blue translucent) showing reach per timestep from start.
* Right  — Lyapunov function V(t) = ‖q(t) − q_goal‖ for both trajectories
  with a sliding-window highlight of the last T/10 seconds.

Usage
-----
Run interactively (opens a matplotlib window)::

    python tools/examples/ppp.py

Save the output image without opening a window::

    python tools/examples/ppp.py --save path/to/output.png
"""

from __future__ import annotations

import argparse
import logging
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
from arco.tools.simulator.scenes.ppp import BOXES as _BOXES
from arco.tools.simulator.scenes.ppp import GOAL as _GOAL
from arco.tools.simulator.scenes.ppp import START as _START
from arco.tools.simulator.scenes.ppp import (
    _sample_box_surface,
)
from arco.tools.simulator.scenes.ppp import is_wall as _is_wall

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Occupancy map
# ---------------------------------------------------------------------------


def build_occupancy(planner_cfg: dict) -> KDTreeOccupancy:
    """Build the 3-D warehouse obstacle map.

    Samples the surface of every box (using the shared
    :func:`~scenes.ppp._sample_box_surface` helper) and wraps the combined
    point cloud in a :class:`~arco.mapping.KDTreeOccupancy`.

    Returns:
        A collision-query-ready 3-D occupancy map.
    """
    all_pts: list[list[float]] = []
    for box in _BOXES:
        all_pts.extend(_sample_box_surface(*box))
    return KDTreeOccupancy(
        all_pts, clearance=float(planner_cfg["obstacle_clearance"])
    )


# ---------------------------------------------------------------------------
# 3-D box renderer (matplotlib)
# ---------------------------------------------------------------------------


def _draw_box(
    ax: plt.Axes,
    x1: float,
    y1: float,
    z1: float,
    x2: float,
    y2: float,
    z2: float,
    color: str = "saddlebrown",
    alpha: float = 0.40,
) -> None:
    """Render an axis-aligned box with ``bar3d``.

    Args:
        ax: Matplotlib 3-D axes.
        x1: Minimum x.
        y1: Minimum y.
        z1: Minimum z.
        x2: Maximum x.
        y2: Maximum y.
        z2: Maximum z.
        color: Face color string.
        alpha: Face transparency in [0, 1].
    """
    ax.bar3d(  # type: ignore[attr-defined]
        x1,
        y1,
        z1,
        x2 - x1,
        y2 - y1,
        z2 - z1,
        color=color,
        alpha=alpha,
        shade=True,
        zsort="average",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cfg: dict, save_path: str | None = None) -> None:
    """Run RRT* and SST on the 3-D PPP warehouse and display the result.

    Args:
        save_path: If provided, save the figure to this path instead of
            opening an interactive window.  The parent directory is created
            automatically if it does not exist.
    """
    if save_path is not None:
        matplotlib.use("Agg")

    env_cfg = cfg.get("environment", cfg)
    planner_cfg = cfg.get("planner", cfg)
    sim_cfg = cfg.get("simulator", cfg)

    bounds = [tuple(b) for b in env_cfg["bounds"]]
    occ = build_occupancy(planner_cfg)

    # --- RRT* ---------------------------------------------------------------
    rrt = RRTPlanner(
        occ,
        bounds=bounds,
        max_sample_count=int(planner_cfg["rrt_max_sample_count"]),
        step_size=planner_cfg["step_size"],
        goal_tolerance=float(planner_cfg["goal_tolerance"]),
        collision_check_count=int(planner_cfg["collision_check_count"]),
        goal_bias=float(planner_cfg["goal_bias"]),
        early_stop=True,
    )
    logger.info("Running RRT* in 3-D …")
    rrt_t0 = time.perf_counter()
    rrt_nodes, _, rrt_path = rrt.get_tree(_START.copy(), _GOAL.copy())
    rrt_elapsed = time.perf_counter() - rrt_t0
    rrt_len = _polyline_length(rrt_path)
    if rrt_path is not None:
        logger.info(
            "RRT*: %d waypoints, length=%.1f m", len(rrt_path), rrt_len
        )
    else:
        logger.warning("RRT*: no path found.")

    # --- SST ----------------------------------------------------------------
    sst = SSTPlanner(
        occ,
        bounds=bounds,
        max_sample_count=int(planner_cfg["sst_max_sample_count"]),
        step_size=planner_cfg["step_size"],
        goal_tolerance=float(planner_cfg["goal_tolerance"]),
        witness_radius=float(planner_cfg["witness_radius"]),
        collision_check_count=int(planner_cfg["collision_check_count"]),
        goal_bias=float(planner_cfg["goal_bias"]),
        early_stop=True,
    )
    logger.info("Running SST in 3-D …")
    sst_t0 = time.perf_counter()
    sst_nodes, _, sst_path = sst.get_tree(_START.copy(), _GOAL.copy())
    sst_elapsed = time.perf_counter() - sst_t0
    sst_len = _polyline_length(sst_path)
    if sst_path is not None:
        logger.info("SST: %d waypoints, length=%.1f m", len(sst_path), sst_len)
    else:
        logger.warning("SST: no path found.")

    # --- Path pruning + trajectory optimization (3-D) ----------------------
    pruner = TrajectoryPruner(
        occ,
        collision_check_count=int(planner_cfg["collision_check_count"]),
    )
    opt = TrajectoryOptimizer(
        occ,
        cruise_speed=float(sim_cfg.get("race_speed", 2.0)),
        weight_time=10.0,
        weight_deviation=1.0,
        weight_velocity=1.0,
        weight_collision=5.0,
        sample_count=1,
        max_iter=200,
    )
    rrt_traj: list[np.ndarray] | None = None
    sst_traj: list[np.ndarray] | None = None
    rrt_traj_dur = 0.0
    sst_traj_dur = 0.0
    rrt_traj_len = 0.0
    sst_traj_len = 0.0
    rrt_opt_status = "not-run"
    sst_opt_status = "not-run"
    rrt_feasible = False
    sst_feasible = False
    rrt_durs: list[float] = []
    sst_durs: list[float] = []
    if rrt_path is not None:
        rrt_path = pruner.prune(rrt_path)
        try:
            res = opt.optimize(rrt_path)
            rrt_traj = res.states
            rrt_durs = list(res.durations) if res.durations else []
            rrt_traj_dur = sum(rrt_durs)
            rrt_traj_len = _polyline_length(res.states)
            rrt_opt_status = (
                f"{res.optimizer_status_code}: {res.optimizer_status_text}"
            )
            rrt_feasible = bool(res.is_feasible)
            logger.info("RRT* trajectory optimized: cost=%.3f", res.cost)
        except Exception:
            logger.exception(
                "RRT* TrajectoryOptimizer failed; skipping overlay."
            )
            rrt_opt_status = "exception"
    if sst_path is not None:
        sst_path = pruner.prune(sst_path)
        try:
            res = opt.optimize(sst_path)
            sst_traj = res.states
            sst_durs = list(res.durations) if res.durations else []
            sst_traj_dur = sum(sst_durs)
            sst_traj_len = _polyline_length(res.states)
            sst_opt_status = (
                f"{res.optimizer_status_code}: {res.optimizer_status_text}"
            )
            sst_feasible = bool(res.is_feasible)
            logger.info("SST trajectory optimized: cost=%.3f", res.cost)
        except Exception:
            logger.exception(
                "SST TrajectoryOptimizer failed; skipping overlay."
            )
            sst_opt_status = "exception"

    def _lyapunov_series(
        traj_states: list[np.ndarray] | None,
        traj_durations: list[float] | None,
        goal: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute cumulative time and V(t) = ||state - goal|| for a trajectory.

        Args:
            traj_states: Sequence of joint-space states.
            traj_durations: Per-segment durations (length = len(states) - 1).
            goal: Goal state array.

        Returns:
            Tuple of ``(times, V)`` arrays.  Both empty if input is too short.
        """
        if traj_states is None or len(traj_states) < 2:
            return np.array([]), np.array([])
        durs = traj_durations or [1.0] * (len(traj_states) - 1)
        times = np.concatenate(
            [[0.0], np.cumsum(durs[: len(traj_states) - 1])]
        )
        V = np.array(
            [float(np.linalg.norm(np.asarray(s) - goal)) for s in traj_states]
        )
        return times, V

    # --- 3-D figure ---------------------------------------------------------
    fig = plt.figure(figsize=(18, 6))
    ax_ws = fig.add_subplot(1, 3, 1, projection="3d")
    ax_cs = fig.add_subplot(1, 3, 2, projection="3d")
    ax_lv = fig.add_subplot(1, 3, 3)

    x_lim = (
        float(env_cfg["bounds"][0][0]),
        float(env_cfg["bounds"][0][1]),
    )
    y_lim = (
        float(env_cfg["bounds"][1][0]),
        float(env_cfg["bounds"][1][1]),
    )
    z_lim = (0.0, float(env_cfg["bounds"][2][1]))

    # ---- ax_ws: Combined 3-D workspace -------------------------------------
    for box in _BOXES:
        _draw_box(ax_ws, *box, color=obstacle_hex(), alpha=0.45)

    if rrt_path is not None and len(rrt_path) >= 2:
        arr = np.array(rrt_path)
        ax_ws.plot(arr[:, 0], arr[:, 1], arr[:, 2],  # type: ignore[attr-defined]
                   color=layer_hex("rrt", "path"), linewidth=1.5, alpha=0.7,
                   label="RRT* path")
    if rrt_traj is not None and len(rrt_traj) >= 2:
        tarr = np.array([[p[0], p[1], p[2]] for p in rrt_traj])
        ax_ws.plot(tarr[:, 0], tarr[:, 1], tarr[:, 2], "o-",  # type: ignore[attr-defined]
                   color=layer_hex("rrt", "trajectory"), linewidth=2.5,
                   markersize=3, alpha=0.9, label="RRT* traj")
    if sst_path is not None and len(sst_path) >= 2:
        arr = np.array(sst_path)
        ax_ws.plot(arr[:, 0], arr[:, 1], arr[:, 2],  # type: ignore[attr-defined]
                   color=layer_hex("sst", "path"), linewidth=1.5, alpha=0.7,
                   label="SST path")
    if sst_traj is not None and len(sst_traj) >= 2:
        tarr = np.array([[p[0], p[1], p[2]] for p in sst_traj])
        ax_ws.plot(tarr[:, 0], tarr[:, 1], tarr[:, 2], "o-",  # type: ignore[attr-defined]
                   color=layer_hex("sst", "trajectory"), linewidth=2.5,
                   markersize=3, alpha=0.9, label="SST traj")

    ax_ws.scatter([_START[0]], [_START[1]], [_START[2]],  # type: ignore[attr-defined]
                  color=annotation_hex(), s=80, zorder=6, label="Start")
    ax_ws.scatter([_GOAL[0]], [_GOAL[1]], [_GOAL[2]],  # type: ignore[attr-defined]
                  color=annotation_hex(), marker="x", linewidths=2,
                  s=80, zorder=6, label="Goal")

    ax_ws.set_xlim(*x_lim)  # type: ignore[attr-defined]
    ax_ws.set_ylim(*y_lim)  # type: ignore[attr-defined]
    ax_ws.set_zlim(*z_lim)  # type: ignore[attr-defined]
    ax_ws.set_xlabel("X (m)")  # type: ignore[attr-defined]
    ax_ws.set_ylabel("Y (m)")  # type: ignore[attr-defined]
    ax_ws.set_zlabel("Z (m)")  # type: ignore[attr-defined]
    ax_ws.set_title("PPP robot — 3-D workspace")  # type: ignore[attr-defined]
    ax_ws.set_box_aspect(  # type: ignore[attr-defined]
        [x_lim[1] - x_lim[0], y_lim[1] - y_lim[0], z_lim[1] - z_lim[0]]
    )
    ax_ws.legend(loc="upper right", fontsize=8)  # type: ignore[attr-defined]
    ax_ws.view_init(elev=25, azim=-50)  # type: ignore[attr-defined]

    metrics_lines = [
        f"RRT* steps/nodes: {max(0, len(rrt_path)-1 if rrt_path else 0)}/{len(rrt_nodes)}",
        f"RRT* time: {_format_clock(rrt_elapsed)} | len: {int(round(rrt_len))} m",
        f"RRT* traj dur: {_format_clock(rrt_traj_dur)} | {rrt_opt_status}",
        f"SST steps/nodes: {max(0, len(sst_path)-1 if sst_path else 0)}/{len(sst_nodes)}",
        f"SST time: {_format_clock(sst_elapsed)} | len: {int(round(sst_len))} m",
        f"SST traj dur: {_format_clock(sst_traj_dur)} | {sst_opt_status}",
    ]
    ax_ws.text2D(  # type: ignore[attr-defined]
        0.02, 0.98, "\n".join(metrics_lines),
        transform=ax_ws.transAxes, va="top", ha="left", fontsize=7,
        color="black",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.80,
              "edgecolor": "none"},
    )

    # ---- ax_cs: C-space = workspace for PPP, annotated with vel. constraints
    for box in _BOXES:
        _draw_box(ax_cs, *box, color=obstacle_hex(), alpha=0.25)

    if rrt_path is not None and len(rrt_path) >= 2:
        arr = np.array(rrt_path)
        ax_cs.plot(arr[:, 0], arr[:, 1], arr[:, 2],  # type: ignore[attr-defined]
                   color=layer_hex("rrt", "path"), linewidth=1.5, alpha=0.8,
                   label="RRT* path")
    if sst_path is not None and len(sst_path) >= 2:
        arr = np.array(sst_path)
        ax_cs.plot(arr[:, 0], arr[:, 1], arr[:, 2],  # type: ignore[attr-defined]
                   color=layer_hex("sst", "path"), linewidth=1.5, alpha=0.8,
                   label="SST path")

    # Velocity constraint box (blue translucent) — reach from start in 1 s
    max_vel = float(sim_cfg.get("max_joint_vel", 3.0))
    sx, sy, sz = float(_START[0]), float(_START[1]), float(_START[2])
    _draw_box(ax_cs, sx - max_vel, sy - max_vel, sz - max_vel,
              sx + max_vel, sy + max_vel, sz + max_vel,
              color="blue", alpha=0.08)

    ax_cs.scatter([_START[0]], [_START[1]], [_START[2]],  # type: ignore[attr-defined]
                  color=annotation_hex(), s=80, zorder=6, label="Start")
    ax_cs.scatter([_GOAL[0]], [_GOAL[1]], [_GOAL[2]],  # type: ignore[attr-defined]
                  color=annotation_hex(), marker="x", linewidths=2,
                  s=80, zorder=6, label="Goal")

    ax_cs.set_xlim(*x_lim)  # type: ignore[attr-defined]
    ax_cs.set_ylim(*y_lim)  # type: ignore[attr-defined]
    ax_cs.set_zlim(*z_lim)  # type: ignore[attr-defined]
    ax_cs.set_xlabel("X (m)")  # type: ignore[attr-defined]
    ax_cs.set_ylabel("Y (m)")  # type: ignore[attr-defined]
    ax_cs.set_zlabel("Z (m)")  # type: ignore[attr-defined]
    ax_cs.set_title("C-space (x m, y m, z m)")  # type: ignore[attr-defined]
    ax_cs.set_box_aspect(  # type: ignore[attr-defined]
        [x_lim[1] - x_lim[0], y_lim[1] - y_lim[0], z_lim[1] - z_lim[0]]
    )
    ax_cs.legend(loc="upper right", fontsize=8)  # type: ignore[attr-defined]
    ax_cs.view_init(elev=25, azim=-50)  # type: ignore[attr-defined]

    # ---- ax_lv: Lyapunov V(t) = ‖q(t) − q_goal‖ ---------------------------
    goal_arr = np.asarray(_GOAL, dtype=float)
    rrt_times, rrt_V = _lyapunov_series(rrt_traj, rrt_durs, goal_arr)
    sst_times, sst_V = _lyapunov_series(sst_traj, sst_durs, goal_arr)

    if len(rrt_times) > 0:
        ax_lv.plot(rrt_times, rrt_V,
                   color=layer_hex("rrt", "trajectory"), linewidth=1.8,
                   label="RRT* V(t)")
        window = (rrt_times[-1] - rrt_times[0]) / 10.0
        ax_lv.axvspan(rrt_times[-1] - window, rrt_times[-1],
                      alpha=0.10, color="gray")
    if len(sst_times) > 0:
        ax_lv.plot(sst_times, sst_V,
                   color=layer_hex("sst", "trajectory"), linewidth=1.8,
                   label="SST V(t)")
        window = (sst_times[-1] - sst_times[0]) / 10.0
        ax_lv.axvspan(sst_times[-1] - window, sst_times[-1],
                      alpha=0.10, color="steelblue")

    ax_lv.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax_lv.set_xlabel("Time (s)")
    ax_lv.set_ylabel("V(t) = ‖q − goal‖ (m)")
    ax_lv.set_title("Lyapunov function")
    ax_lv.legend(loc="upper right", fontsize=7)
    ax_lv.grid(True, alpha=0.3)

    plt.suptitle(
        "PPP robot — RRT* vs SST in 3-D warehouse",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info("Saved PPP planning example to %s", save_path)
    else:
        plt.show()


if __name__ == "__main__":
    import yaml as _yaml

    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scenario", metavar="FILE", help="Path to scenario YAML file."
    )
    parser.add_argument(
        "--save",
        metavar="PATH",
        default=None,
        help="Save the figure instead of opening an interactive window.",
    )
    args = parser.parse_args()
    with open(args.scenario) as _fh:
        _cfg = _yaml.safe_load(_fh) or {}
    main(_cfg, save_path=args.save)
