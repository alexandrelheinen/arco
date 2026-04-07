"""2-D RR robot arm planning — RRT* vs SST in joint space.

Runs RRT* and SST planners side-by-side for a two-link planar arm
navigating from a start to a goal configuration while avoiding a
rectangular obstacle.  Planning is done in joint space (theta1, theta2);
resulting paths are converted to Cartesian space via forward kinematics
for visualization.

Figure layout (3 subplots in a row)
------------------------------------
* Left  — RRT* end-effector trace in Cartesian workspace
* Middle — SST end-effector trace in Cartesian workspace
* Right  — Joint C-space: collision configs (gray scatter), both joint
  paths, optimized trajectories, start / goal markers

Usage
-----
Run interactively (opens a matplotlib window)::

    python tools/examples/rr_planning.py

Save the output image without opening a window::

    python tools/examples/rr_planning.py --save path/to/output.png
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulator"))

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from logging_config import configure_logging
from scenes.rr import (
    build_cspace_occupancy,
    pick_collision_free_ik,
)

from arco.kinematics import RRRobot
from arco.planning.continuous import (
    RRTPlanner,
    SSTPlanner,
    TrajectoryOptimizer,
)
from config import load_config

logger = logging.getLogger(__name__)

_cfg = load_config("rr")

# Minimum annulus inner radius below which the inner-hole outline is skipped.
_INNER_RADIUS_THRESHOLD: float = 1e-6

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
# Visualization helpers
# ---------------------------------------------------------------------------


def _draw_arm(
    ax: plt.Axes,
    robot: RRRobot,
    q1: float,
    q2: float,
    color: str = "navy",
    alpha: float = 0.6,
    label: str | None = None,
) -> None:
    """Draw the two-link arm on Cartesian axes.

    Args:
        ax: Matplotlib axes.
        robot: The :class:`~arco.kinematics.RRRobot` instance.
        q1: First joint angle in radians.
        q2: Second joint angle in radians.
        color: Line and marker color.
        alpha: Transparency in [0, 1].
        label: Optional legend label.
    """
    origin, j2, ee = robot.link_segments(q1, q2)
    xs = [origin[0], j2[0], ee[0]]
    ys = [origin[1], j2[1], ee[1]]
    ax.plot(xs, ys, "-o", color=color, linewidth=2, alpha=alpha, label=label)


def _fk_path(
    robot: RRRobot, joint_path: list[np.ndarray] | None
) -> list[tuple[float, float]] | None:
    """Convert a joint-space path to Cartesian end-effector positions.

    Args:
        robot: The :class:`~arco.kinematics.RRRobot` instance.
        joint_path: List of 2-element arrays ``[q1, q2]``, or ``None``.

    Returns:
        List of ``(x, y)`` tuples, or ``None`` if *joint_path* is ``None``.
    """
    if joint_path is None:
        return None
    return [
        robot.forward_kinematics(float(pt[0]), float(pt[1]))
        for pt in joint_path
    ]


# ---------------------------------------------------------------------------
# Main planning function
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the RR planning example and display or save the figure."""
    configure_logging()
    parser = argparse.ArgumentParser(
        description="RR robot arm planning example"
    )
    parser.add_argument(
        "--save",
        metavar="PATH",
        default="",
        help="Save the figure to this path instead of showing it.",
    )
    args = parser.parse_args()

    if args.save:
        matplotlib.use("Agg")

    # --- Setup -----------------------------------------------------------
    robot = RRRobot(l1=float(_cfg["l1"]), l2=float(_cfg["l2"]))
    obstacles: list[list[float]] = [
        [float(v) for v in obs] for obs in _cfg["obstacles"]
    ]
    bounds = [tuple(b) for b in _cfg["bounds"]]
    clearance = float(_cfg["obstacle_clearance"])

    start_xy = [float(v) for v in _cfg["start_xy"]]
    goal_xy = [float(v) for v in _cfg["goal_xy"]]

    start_q = pick_collision_free_ik(robot, start_xy, obstacles, [-2.2, 1.8])
    goal_q = pick_collision_free_ik(robot, goal_xy, obstacles, [1.0, -1.6])

    logger.info("Start joint config: (%.3f, %.3f)", start_q[0], start_q[1])
    logger.info("Goal  joint config: (%.3f, %.3f)", goal_q[0], goal_q[1])

    # --- Occupancy map ---------------------------------------------------
    logger.info("Building C-space occupancy map …")
    occ, collision_pts = build_cspace_occupancy(robot, obstacles, clearance)

    # --- RRT* ------------------------------------------------------------
    rrt = RRTPlanner(
        occ,
        bounds=bounds,
        max_sample_count=int(_cfg["rrt_max_sample_count"]),
        step_size=float(_cfg["step_size"]),
        goal_tolerance=float(_cfg["goal_tolerance"]),
        collision_check_count=int(_cfg["collision_check_count"]),
        goal_bias=float(_cfg["goal_bias"]),
        early_stop=True,
    )
    logger.info("Running RRT* in joint space …")
    rrt_t0 = time.perf_counter()
    rrt_result = rrt.plan(start_q, goal_q)
    rrt_elapsed = time.perf_counter() - rrt_t0
    rrt_path: list[np.ndarray] | None = (
        [np.asarray(p) for p in rrt_result] if rrt_result else None
    )
    rrt_feasible = rrt_path is not None
    rrt_nodes: list = list(getattr(rrt, "_nodes", []))
    rrt_len = _polyline_length(rrt_path)
    logger.info(
        "RRT*: %d waypoints, length=%.3f rad",
        len(rrt_path) if rrt_path else 0,
        rrt_len,
    )

    # --- SST -------------------------------------------------------------
    sst = SSTPlanner(
        occ,
        bounds=bounds,
        max_sample_count=int(_cfg["sst_max_sample_count"]),
        step_size=float(_cfg["step_size"]),
        goal_tolerance=float(_cfg["goal_tolerance"]),
        collision_check_count=int(_cfg["collision_check_count"]),
        goal_bias=float(_cfg["goal_bias"]),
        witness_radius=float(_cfg["witness_radius"]),
        early_stop=True,
    )
    logger.info("Running SST in joint space …")
    sst_t0 = time.perf_counter()
    sst_result = sst.plan(start_q, goal_q)
    sst_elapsed = time.perf_counter() - sst_t0
    sst_path: list[np.ndarray] | None = (
        [np.asarray(p) for p in sst_result] if sst_result else None
    )
    sst_feasible = sst_path is not None
    sst_nodes: list = list(getattr(sst, "_nodes", []))
    sst_len = _polyline_length(sst_path)
    logger.info(
        "SST: %d waypoints, length=%.3f rad",
        len(sst_path) if sst_path else 0,
        sst_len,
    )

    # --- Trajectory optimization -----------------------------------------
    optimizer = TrajectoryOptimizer(
        occ,
        cruise_speed=float(_cfg.get("race_speed", 1.0)),
        weight_time=10.0,
        weight_deviation=1.0,
        weight_velocity=1.0,
        weight_collision=5.0,
        sample_count=1,
        max_iter=200,
    )

    def _optimize(
        path: list[np.ndarray] | None,
        label: str,
    ) -> tuple[list[np.ndarray] | None, float, float, str]:
        if path is None or len(path) < 2:
            return None, 0.0, 0.0, "no-path"
        try:
            result = optimizer.optimize(path)
            traj = list(result.states)
            dur = float(sum(result.durations)) if result.durations else 0.0
            status = (
                f"{result.optimizer_status_code}:"
                f" {result.optimizer_status_text}"
            )
            logger.info(
                "%s trajectory optimized: cost=%.3f", label, result.cost
            )
        except Exception as exc:
            logger.warning("%s optimization failed: %s", label, exc)
            traj = list(path)
            dur = 0.0
            status = "error"
        return traj, _polyline_length(traj), dur, status

    rrt_traj, rrt_traj_len, rrt_traj_dur, rrt_opt_status = _optimize(
        rrt_path, "RRT*"
    )
    sst_traj, sst_traj_len, sst_traj_dur, sst_opt_status = _optimize(
        sst_path, "SST"
    )

    # --- Workspace geometry ----------------------------------------------
    r_min, r_max = robot.workspace_annulus()
    theta_range = np.linspace(0, 2 * math.pi, 200)
    outer_x = r_max * np.cos(theta_range)
    outer_y = r_max * np.sin(theta_range)
    inner_x = r_min * np.cos(theta_range)
    inner_y = r_min * np.sin(theta_range)

    def _obs_patch(obs: list[float]) -> mpatches.Rectangle:
        xmin, ymin, xmax, ymax = obs
        return mpatches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=1,
            edgecolor="red",
            facecolor="tomato",
            alpha=0.5,
            label="Obstacle",
        )

    rrt_cart = _fk_path(robot, rrt_path)
    sst_cart = _fk_path(robot, sst_path)
    rrt_traj_cart = _fk_path(robot, rrt_traj)
    sst_traj_cart = _fk_path(robot, sst_traj)

    # --- Figure ----------------------------------------------------------
    fig, (ax_rrt, ax_sst, ax_joint) = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#0d1117")
    for ax in (ax_rrt, ax_sst, ax_joint):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    specs = [
        (
            ax_rrt,
            "RRT* — RR robot Cartesian space",
            rrt_cart,
            rrt_traj_cart,
            [_obs_patch(obs) for obs in obstacles],
            "royalblue",
            rrt_path,
            {
                "steps": max(0, (len(rrt_path) - 1) if rrt_path else 0),
                "nodes": len(rrt_nodes),
                "planner_time": rrt_elapsed,
                "path_len": rrt_len,
                "traj_len": rrt_traj_len,
                "traj_dur": rrt_traj_dur,
                "path_status": ("found" if rrt_feasible else "stalled"),
                "opt_status": rrt_opt_status,
            },
        ),
        (
            ax_sst,
            "SST — RR robot Cartesian space",
            sst_cart,
            sst_traj_cart,
            [_obs_patch(obs) for obs in obstacles],
            "mediumseagreen",
            sst_path,
            {
                "steps": max(0, (len(sst_path) - 1) if sst_path else 0),
                "nodes": len(sst_nodes),
                "planner_time": sst_elapsed,
                "path_len": sst_len,
                "traj_len": sst_traj_len,
                "traj_dur": sst_traj_dur,
                "path_status": ("found" if sst_feasible else "stalled"),
                "opt_status": sst_opt_status,
            },
        ),
    ]

    for (
        ax,
        title,
        cart_path,
        traj_cart,
        obs_patches,
        color,
        joint_path,
        metrics,
    ) in specs:
        # Workspace annulus
        ax.fill(outer_x, outer_y, alpha=0.07, color="steelblue")
        if r_min > _INNER_RADIUS_THRESHOLD:
            ax.fill(inner_x, inner_y, alpha=0.25, color="#161b22")
        ax.plot(outer_x, outer_y, color="steelblue", linewidth=0.8, alpha=0.5)
        if r_min > _INNER_RADIUS_THRESHOLD:
            ax.plot(
                inner_x,
                inner_y,
                color="steelblue",
                linewidth=0.8,
                alpha=0.5,
            )

        # Obstacles
        for obs_patch in obs_patches:
            ax.add_patch(obs_patch)

        # Arm at start and goal
        _draw_arm(
            ax,
            robot,
            float(start_q[0]),
            float(start_q[1]),
            color="limegreen",
            alpha=0.8,
            label="Start arm",
        )
        _draw_arm(
            ax,
            robot,
            float(goal_q[0]),
            float(goal_q[1]),
            color="orangered",
            alpha=0.8,
            label="Goal arm",
        )

        # Planned path (end-effector trace)
        if cart_path is not None and len(cart_path) >= 2:
            arr = np.array(cart_path)
            path_alpha = 0.35 if traj_cart is not None else 0.9
            ax.plot(
                arr[:, 0],
                arr[:, 1],
                color=color,
                linewidth=1.5,
                alpha=path_alpha,
                zorder=5,
                label=f"Path ({_polyline_length(joint_path):.2f} rad)",
            )

        # Optimized trajectory overlay
        if traj_cart is not None and len(traj_cart) >= 2:
            tarr = np.array(traj_cart)
            ax.plot(
                tarr[:, 0],
                tarr[:, 1],
                "o-",
                color="orangered",
                linewidth=2.0,
                markersize=3,
                zorder=7,
                alpha=0.9,
                label="Optimized traj",
            )

        # Start / goal end-effector markers
        sx, sy = robot.forward_kinematics(float(start_q[0]), float(start_q[1]))
        gx, gy = robot.forward_kinematics(float(goal_q[0]), float(goal_q[1]))
        ax.plot(sx, sy, "o", color="limegreen", ms=8, zorder=9)
        ax.plot(gx, gy, "*", color="orangered", ms=12, zorder=9)

        ax.set_aspect("equal")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=7)

        metrics_lines = [
            f"Steps/nodes: {metrics['steps']}/{metrics['nodes']}",
            f"Planner time: {_format_clock(float(metrics['planner_time']))}",
            f"Path length: {float(metrics['path_len']):.3f} rad",
            f"Traj length: {float(metrics['traj_len']):.3f} rad",
            f"Traj duration: {_format_clock(float(metrics['traj_dur']))}",
            f"Status: {metrics['path_status']}",
            f"Optimizer: {metrics['opt_status']}",
        ]
        ax.text(
            0.02,
            0.98,
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
        )

    # --- Joint space subplot -------------------------------------------
    if collision_pts:
        cpts = np.array(collision_pts)
        ax_joint.scatter(
            cpts[:, 0],
            cpts[:, 1],
            s=1,
            c="gray",
            alpha=0.3,
            label="C-space obstacle",
        )

    if rrt_path is not None:
        arr = np.array(rrt_path)
        ax_joint.plot(
            arr[:, 0],
            arr[:, 1],
            color="royalblue",
            linewidth=1.5,
            alpha=0.8,
            label="RRT* path",
        )
    if rrt_traj is not None:
        tarr = np.array(rrt_traj)
        ax_joint.plot(
            tarr[:, 0],
            tarr[:, 1],
            "o-",
            color="cornflowerblue",
            linewidth=2.0,
            markersize=3,
            alpha=0.9,
            label="RRT* traj",
        )

    if sst_path is not None:
        arr = np.array(sst_path)
        ax_joint.plot(
            arr[:, 0],
            arr[:, 1],
            color="mediumseagreen",
            linewidth=1.5,
            alpha=0.8,
            label="SST path",
        )
    if sst_traj is not None:
        tarr = np.array(sst_traj)
        ax_joint.plot(
            tarr[:, 0],
            tarr[:, 1],
            "o-",
            color="limegreen",
            linewidth=2.0,
            markersize=3,
            alpha=0.9,
            label="SST traj",
        )

    # Start / goal markers in C-space
    ax_joint.plot(
        float(start_q[0]),
        float(start_q[1]),
        "o",
        color="limegreen",
        ms=10,
        zorder=9,
        label="Start",
    )
    ax_joint.plot(
        float(goal_q[0]),
        float(goal_q[1]),
        "*",
        color="orangered",
        ms=14,
        zorder=9,
        label="Goal",
    )

    ax_joint.set_xlim(-math.pi, math.pi)
    ax_joint.set_ylim(-math.pi, math.pi)
    ax_joint.set_xlabel("θ₁ (rad)")
    ax_joint.set_ylabel("θ₂ (rad)")
    ax_joint.set_title("Joint C-space — configuration trajectories")
    ax_joint.legend(loc="upper right", fontsize=7)

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=120, bbox_inches="tight")
        logger.info("Saved RR planning example to %s", args.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()
