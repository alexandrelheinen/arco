"""PPP (triple prismatic) robot planning in a 3-D warehouse environment.

Runs RRT* and SST planners on an industrial warehouse bay modelled as a
box-shaped work volume (60 m × 20 m × 6 m) with ground-level box obstacles.
Three width-crossing barriers increase difficulty.

For PPP robots the C-space is the workspace (all joints are prismatic), so
the C-space panel annotates the same 3-D volume with velocity constraints.

Figure layout (standard two-frame)
------------------------------------
* Top-left  — Workspace: both RRT* and SST paths/trajectories in 3-D.
* Top-right — C-space (x m, y m, z m): same volume, annotated with the
  velocity constraint box (blue translucent) showing reach per timestep
  from start.
* Bottom    — Metrics: per-planner step counts, planning times, path and
  trajectory lengths, optimiser status.

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

from arco.config.palette import obstacle_hex
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
from arco.tools.viewer import FrameRenderer, SceneSnapshot
from arco.tools.viewer.layout import StandardLayout
from arco.tools.viewer.utils import format_clock, polyline_length

logger = logging.getLogger(__name__)


def _build_ppp_snapshot(
    planner: str,
    start: np.ndarray,
    goal: np.ndarray,
    path: list[np.ndarray] | None,
    traj: list[np.ndarray] | None,
) -> SceneSnapshot:
    """Build a 3-D SceneSnapshot for the PPP example (workspace = C-space).

    Args:
        planner: Planner key, e.g. ``"rrt"`` or ``"sst"``.
        start: Start position ``[x, y, z]``.
        goal: Goal position ``[x, y, z]``.
        path: Raw planned path.
        traj: Optimised trajectory states.

    Returns:
        A :class:`~arco.tools.viewer.SceneSnapshot` in 3-D Cartesian space.
    """
    return SceneSnapshot.from_planning_result(
        scenario="ppp",
        planner=planner,
        start=[float(start[0]), float(start[1]), float(start[2])],
        goal=[float(goal[0]), float(goal[1]), float(goal[2])],
        obstacles=[],  # drawn as _draw_box() manually
        found_path=(
            [[float(p[0]), float(p[1]), float(p[2])] for p in path]
            if path
            else None
        ),
        adjusted_trajectory=(
            [[float(p[0]), float(p[1]), float(p[2])] for p in traj]
            if traj
            else None
        ),
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
        cfg: Scenario configuration dictionary (loaded from ``ppp.yml``).
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
    rrt_len = polyline_length(rrt_path)
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
    sst_len = polyline_length(sst_path)
    if sst_path is not None:
        logger.info("SST: %d waypoints, length=%.1f m", len(sst_path), sst_len)
    else:
        logger.warning("SST: no path found.")

    # --- Path pruning + trajectory optimisation (3-D) ----------------------
    pruner = TrajectoryPruner(
        occ,
        step_size=np.asarray(planner_cfg["step_size"], dtype=float),
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
    if rrt_path is not None:
        rrt_path = pruner.prune(rrt_path)
        try:
            res = opt.optimize(rrt_path)
            rrt_traj = res.states
            rrt_traj_dur = sum(res.durations) if res.durations else 0.0
            rrt_traj_len = polyline_length(res.states)
            rrt_opt_status = (
                f"{res.optimizer_status_code}: {res.optimizer_status_text}"
            )
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
            sst_traj_dur = sum(res.durations) if res.durations else 0.0
            sst_traj_len = polyline_length(res.states)
            sst_opt_status = (
                f"{res.optimizer_status_code}: {res.optimizer_status_text}"
            )
            logger.info("SST trajectory optimized: cost=%.3f", res.cost)
        except Exception:
            logger.exception(
                "SST TrajectoryOptimizer failed; skipping overlay."
            )
            sst_opt_status = "exception"

    rrt_snap = _build_ppp_snapshot(
        "rrt", np.array(_START), np.array(_GOAL), rrt_path, rrt_traj
    )
    sst_snap = _build_ppp_snapshot(
        "sst", np.array(_START), np.array(_GOAL), sst_path, sst_traj
    )

    # --- 3-D figure ---------------------------------------------------------
    fig, ax_ws, ax_cs, ax_bottom = StandardLayout.create(
        title="PPP robot — RRT* vs SST in 3-D warehouse",
        ws_3d=True,
        cs_3d=True,
    )

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

    FrameRenderer(draw_tree=False, draw_obstacles=False, is_3d=True).render(
        ax_ws, rrt_snap
    )
    FrameRenderer(
        draw_tree=False,
        draw_obstacles=False,
        draw_start_goal=False,
        is_3d=True,
    ).render(ax_ws, sst_snap)

    ax_ws.set_xlim(*x_lim)  # type: ignore[attr-defined]
    ax_ws.set_ylim(*y_lim)  # type: ignore[attr-defined]
    ax_ws.set_zlim(*z_lim)  # type: ignore[attr-defined]
    ax_ws.set_xlabel("X (m)")  # type: ignore[attr-defined]
    ax_ws.set_ylabel("Y (m)")  # type: ignore[attr-defined]
    ax_ws.set_zlabel("Z (m)")  # type: ignore[attr-defined]
    ax_ws.set_title("Workspace")  # type: ignore[attr-defined]
    ax_ws.set_box_aspect(  # type: ignore[attr-defined]
        [x_lim[1] - x_lim[0], y_lim[1] - y_lim[0], z_lim[1] - z_lim[0]]
    )
    ax_ws.legend(loc="upper right", fontsize=8)  # type: ignore[attr-defined]
    ax_ws.view_init(elev=25, azim=-50)  # type: ignore[attr-defined]

    # ---- ax_cs: C-space = workspace for PPP, annotated with vel. constraints
    for box in _BOXES:
        _draw_box(ax_cs, *box, color=obstacle_hex(), alpha=0.25)

    if rrt_path is not None and len(rrt_path) >= 2:
        arr = np.array(rrt_path)
        ax_cs.plot(
            arr[:, 0],
            arr[:, 1],
            arr[:, 2],  # type: ignore[attr-defined]
            color=layer_hex("rrt", "path"),
            linewidth=1.5,
            alpha=0.8,
            label="RRT* path",
        )
    if sst_path is not None and len(sst_path) >= 2:
        arr = np.array(sst_path)
        ax_cs.plot(
            arr[:, 0],
            arr[:, 1],
            arr[:, 2],  # type: ignore[attr-defined]
            color=layer_hex("sst", "path"),
            linewidth=1.5,
            alpha=0.8,
            label="SST path",
        )

    max_vel = float(sim_cfg.get("max_joint_vel", 3.0))
    sx, sy, sz = float(_START[0]), float(_START[1]), float(_START[2])
    _draw_box(
        ax_cs,
        sx - max_vel,
        sy - max_vel,
        sz - max_vel,
        sx + max_vel,
        sy + max_vel,
        sz + max_vel,
        color="blue",
        alpha=0.08,
    )

    ax_cs.scatter(
        [_START[0]],
        [_START[1]],
        [_START[2]],  # type: ignore[attr-defined]
        color=annotation_hex(),
        s=80,
        zorder=6,
        label="Start",
    )
    ax_cs.scatter(
        [_GOAL[0]],
        [_GOAL[1]],
        [_GOAL[2]],  # type: ignore[attr-defined]
        color=annotation_hex(),
        marker="x",
        linewidths=2,
        s=80,
        zorder=6,
        label="Goal",
    )

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

    # ---- Bottom: metrics ---------------------------------------------------
    StandardLayout.write_metrics(
        ax_bottom,
        [
            f"RRT*  steps/nodes: "
            f"{max(0, len(rrt_path)-1 if rrt_path else 0)}/{len(rrt_nodes)} | "
            f"time: {format_clock(rrt_elapsed)} | "
            f"path: {int(round(rrt_len))} m | "
            f"traj: {format_clock(rrt_traj_dur)} | {rrt_opt_status}",
            f"SST   steps/nodes: "
            f"{max(0, len(sst_path)-1 if sst_path else 0)}/{len(sst_nodes)} | "
            f"time: {format_clock(sst_elapsed)} | "
            f"path: {int(round(sst_len))} m | "
            f"traj: {format_clock(sst_traj_dur)} | {sst_opt_status}",
        ],
    )

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
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
