"""PPP (triple prismatic) robot planning in a 3-D warehouse environment.

Runs RRT* and SST planners side-by-side on an industrial warehouse bay
modelled as a box-shaped work volume (60 m × 20 m × 6 m) with ground-level
box obstacles. Three width-crossing barriers increase difficulty: a tall
barrier, a smaller one, then a split-half barrier with mixed heights.

Only the chosen path is rendered — no exploration tree — to keep the 3-D
view uncluttered.

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
from arco.tools.config import load_config
from arco.tools.logging_config import configure_logging
from arco.tools.simulator.scenes.ppp import BOXES as _BOXES
from arco.tools.simulator.scenes.ppp import GOAL as _GOAL
from arco.tools.simulator.scenes.ppp import START as _START
from arco.tools.simulator.scenes.ppp import (
    _sample_box_surface,
)
from arco.tools.simulator.scenes.ppp import is_wall as _is_wall

logger = logging.getLogger(__name__)

_cfg = load_config("ppp")
_env_cfg = _cfg.get("environment", _cfg)
_planner_cfg = _cfg.get("planner", _cfg)
_sim_cfg = _cfg.get("simulator", _cfg)


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


def build_occupancy() -> KDTreeOccupancy:
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
        all_pts, clearance=float(_planner_cfg["obstacle_clearance"])
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


def main(save_path: str | None = None) -> None:
    """Run RRT* and SST on the 3-D PPP warehouse and display the result.

    Args:
        save_path: If provided, save the figure to this path instead of
            opening an interactive window.  The parent directory is created
            automatically if it does not exist.
    """
    if save_path is not None:
        matplotlib.use("Agg")

    bounds = [tuple(b) for b in _env_cfg["bounds"]]
    occ = build_occupancy()

    # --- RRT* ---------------------------------------------------------------
    rrt = RRTPlanner(
        occ,
        bounds=bounds,
        max_sample_count=int(_planner_cfg["rrt_max_sample_count"]),
        step_size=float(_planner_cfg["step_size"]),
        goal_tolerance=float(_planner_cfg["goal_tolerance"]),
        collision_check_count=int(_planner_cfg["collision_check_count"]),
        goal_bias=float(_planner_cfg["goal_bias"]),
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
        max_sample_count=int(_planner_cfg["sst_max_sample_count"]),
        step_size=float(_planner_cfg["step_size"]),
        goal_tolerance=float(_planner_cfg["goal_tolerance"]),
        witness_radius=float(_planner_cfg["witness_radius"]),
        collision_check_count=int(_planner_cfg["collision_check_count"]),
        goal_bias=float(_planner_cfg["goal_bias"]),
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

    # --- Trajectory optimization (3-D) -------------------------------------
    opt = TrajectoryOptimizer(
        occ,
        cruise_speed=float(_sim_cfg.get("race_speed", 2.0)),
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
    if rrt_path is not None:
        try:
            res = opt.optimize(rrt_path)
            rrt_traj = res.states
            rrt_traj_dur = sum(res.durations)
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
        try:
            res = opt.optimize(sst_path)
            sst_traj = res.states
            sst_traj_dur = sum(res.durations)
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

    # --- 3-D figure ---------------------------------------------------------
    fig = plt.figure(figsize=(14, 7))

    specs = [
        (
            "RRT* — 3-D PPP warehouse",
            rrt_path,
            rrt_len,
            "royalblue",
            rrt_traj,
            {
                "steps": max(0, (len(rrt_path) - 1) if rrt_path else 0),
                "nodes": len(rrt_nodes),
                "planner_time": rrt_elapsed,
                "path_len": rrt_len,
                "traj_len": rrt_traj_len,
                "traj_dur": rrt_traj_dur,
                "path_status": (
                    "found"
                    if (rrt_path is not None and rrt_feasible)
                    else "stalled"
                ),
                "opt_status": rrt_opt_status,
            },
        ),
        (
            "SST — 3-D PPP warehouse",
            sst_path,
            sst_len,
            "mediumseagreen",
            sst_traj,
            {
                "steps": max(0, (len(sst_path) - 1) if sst_path else 0),
                "nodes": len(sst_nodes),
                "planner_time": sst_elapsed,
                "path_len": sst_len,
                "traj_len": sst_traj_len,
                "traj_dur": sst_traj_dur,
                "path_status": (
                    "found"
                    if (sst_path is not None and sst_feasible)
                    else "stalled"
                ),
                "opt_status": sst_opt_status,
            },
        ),
    ]
    x_lim = (
        float(_env_cfg["bounds"][0][0]),
        float(_env_cfg["bounds"][0][1]),
    )
    y_lim = (
        float(_env_cfg["bounds"][1][0]),
        float(_env_cfg["bounds"][1][1]),
    )
    z_lim = (0.0, float(_env_cfg["bounds"][2][1]))

    for col, (title, path, length, color, traj, metrics) in enumerate(specs):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")

        # Obstacle boxes — wall is colored distinctly from scatter boxes.
        for box in _BOXES:
            _draw_box(
                ax,
                *box,
                color="sienna" if _is_wall(box) else "peru",
                alpha=0.45,
            )

        # Solution path — dimmed when trajectory is drawn on top
        if path is not None and len(path) >= 2:
            arr = np.array(path)
            path_alpha = 0.35 if traj is not None else 1.0
            label = f"Path  {length:.1f} m | {len(path)} wpts"
            ax.plot(  # type: ignore[attr-defined]
                arr[:, 0],
                arr[:, 1],
                arr[:, 2],
                color=color,
                linewidth=1.5,
                alpha=path_alpha,
                zorder=5,
                label=label,
            )

        # Optimized trajectory — bright highlight on top of path
        if traj is not None and len(traj) >= 2:
            tarr = np.array([[p[0], p[1], p[2]] for p in traj])
            ax.plot(  # type: ignore[attr-defined]
                tarr[:, 0],
                tarr[:, 1],
                tarr[:, 2],
                "o-",
                color="orangered",
                linewidth=2.5,
                markersize=3,
                zorder=7,
                label="Optimized trajectory",
            )

        # Start and goal markers
        ax.scatter(  # type: ignore[attr-defined]
            [_START[0]],
            [_START[1]],
            [_START[2]],
            color="limegreen",
            s=80,
            zorder=6,
            label="Start",
        )
        ax.scatter(  # type: ignore[attr-defined]
            [_GOAL[0]],
            [_GOAL[1]],
            [_GOAL[2]],
            color="orangered",
            marker="*",
            s=120,
            zorder=6,
            label="Goal",
        )

        ax.set_xlim(*x_lim)  # type: ignore[attr-defined]
        ax.set_ylim(*y_lim)  # type: ignore[attr-defined]
        ax.set_zlim(*z_lim)  # type: ignore[attr-defined]
        ax.set_xlabel("X (m)")  # type: ignore[attr-defined]
        ax.set_ylabel("Y (m)")  # type: ignore[attr-defined]
        ax.set_zlabel("Z (m)")  # type: ignore[attr-defined]
        ax.set_title(title)  # type: ignore[attr-defined]
        ax.set_box_aspect(  # type: ignore[attr-defined]
            [x_lim[1] - x_lim[0], y_lim[1] - y_lim[0], z_lim[1] - z_lim[0]]
        )
        ax.legend(loc="upper right", fontsize=8)  # type: ignore[attr-defined]
        ax.view_init(elev=25, azim=-50)  # type: ignore[attr-defined]
        metrics_lines = [
            (
                f"Planner steps / nodes: "
                f"{metrics['steps']} / {metrics['nodes']}"
            ),
            f"Planner time: {_format_clock(float(metrics['planner_time']))}",
            (
                f"Planned path length: "
                f"{int(round(float(metrics['path_len'])))} m"
            ),
            (
                f"Trajectory arc length: "
                f"{int(round(float(metrics['traj_len'])))} m"
            ),
            (
                f"Trajectory duration: "
                f"{_format_clock(float(metrics['traj_dur']))}"
            ),
            f"Path status: {metrics['path_status']}",
            f"Optimizer status: {metrics['opt_status']}",
        ]
        ax.text2D(  # type: ignore[attr-defined]
            0.02,
            0.98,
            "\n".join(metrics_lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7,
            color="black",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "alpha": 0.80,
                "edgecolor": "none",
            },
        )

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
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save",
        metavar="PATH",
        default=None,
        help="Save the figure instead of opening an interactive window.",
    )
    args = parser.parse_args()
    main(save_path=args.save)
