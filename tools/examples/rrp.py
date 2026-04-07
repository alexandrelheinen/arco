"""RRP SCARA-like arm planning — RRT* vs SST in 3-D joint space.

Runs RRT* and SST planners side-by-side for a two-link planar arm with a
vertical prismatic joint (RRP / SCARA-like).  The cylindrical workspace
contains two pillar obstacles (forcing XY routing) and two slab obstacles
(forcing Z routing), so a plan must combine all three joint degrees of
freedom.

Figure layout (3 subplots in a row)
-------------------------------------
* Left   — RRT* end-effector trace in 3-D Cartesian workspace
* Middle — SST end-effector trace in 3-D Cartesian workspace
* Right  — C-space (q1, q2) scatter coloured by collision-config Z height

Usage
-----
Run interactively (opens a matplotlib window)::

    python tools/examples/rrp.py

Save the output image without opening a window::

    python tools/examples/rrp.py --save path/to/output.png
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
import matplotlib.pyplot as plt
import numpy as np
from logging_config import configure_logging
from scenes.rrp import (
    _arm_collides_3d,
    build_cspace_occupancy_3d,
    pick_collision_free_config,
)

from arco.kinematics import RRPRobot
from arco.planning.continuous import (
    RRTPlanner,
    SSTPlanner,
    TrajectoryOptimizer,
)
from config import load_config

logger = logging.getLogger(__name__)

_cfg = load_config("rrp")
_robot_cfg = _cfg.get("robot", _cfg)
_env_cfg = _cfg.get("environment", _cfg)
_planner_cfg = _cfg.get("planner", _cfg)
_sim_cfg = _cfg.get("simulator", _cfg)


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


def _fk_path(
    robot: RRPRobot, joint_path: list[np.ndarray] | None
) -> list[tuple[float, float, float]] | None:
    """Convert a 3-D joint-space path to Cartesian end-effector positions.

    Args:
        robot: The :class:`~arco.kinematics.RRPRobot` instance.
        joint_path: List of ``[q1, q2, z]`` arrays, or ``None``.

    Returns:
        List of ``(x, y, z)`` tuples, or ``None`` if *joint_path* is ``None``.
    """
    if joint_path is None:
        return None
    return [
        robot.forward_kinematics(float(pt[0]), float(pt[1]), float(pt[2]))
        for pt in joint_path
    ]


# ---------------------------------------------------------------------------
# Box drawing
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
    alpha: float = 0.35,
) -> None:
    """Render an axis-aligned 3-D box with ``bar3d``.

    Args:
        ax: Matplotlib 3-D axes.
        x1-z2: Box extents.
        color: Face color.
        alpha: Face transparency.
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


def _workspace_annulus(
    ax: plt.Axes,
    r_min: float,
    r_max: float,
    z_min: float,
    z_max: float,
    segments: int = 60,
) -> None:
    """Draw dashed annulus outline at *z_min* for workspace reference.

    Args:
        ax: Matplotlib 3-D axes.
        r_min: Inner radius of annulus.
        r_max: Outer radius of annulus.
        z_min: Z level for the ring.
        z_max: Top Z level for cylinder wall lines.
        segments: Number of polygon vertices.
    """
    angles = np.linspace(0, 2 * math.pi, segments + 1)
    for r in (r_min, r_max):
        xs = r * np.cos(angles)
        ys = r * np.sin(angles)
        zs = np.full_like(xs, z_min)
        ax.plot(  # type: ignore[attr-defined]
            xs, ys, zs, color="gray", linewidth=0.6, alpha=0.4, linestyle="--"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(save_path: str | None = None) -> None:
    """Run RRT* and SST on the 3-D RRP arm and display the result.

    Args:
        save_path: If provided, save the figure to this path instead of
            opening an interactive window.  The parent directory is created
            automatically if it does not exist.
    """
    if save_path is not None:
        matplotlib.use("Agg")

    robot = RRPRobot(
        l1=float(_robot_cfg["l1"]),
        l2=float(_robot_cfg["l2"]),
        z_min=float(_robot_cfg["z_min"]),
        z_max=float(_robot_cfg["z_max"]),
    )
    obstacles: list[list[float]] = [
        [float(v) for v in obs] for obs in _env_cfg["obstacles"]
    ]
    bounds = [tuple(b) for b in _env_cfg["bounds"]]
    clearance = float(_env_cfg["obstacle_clearance"])
    start_position = [float(v) for v in _env_cfg["start_position"]]
    goal_position = [float(v) for v in _env_cfg["goal_position"]]
    if len(start_position) >= 3:
        start_xy = start_position[:2]
        start_z = float(start_position[2])
    else:
        # Backward compatibility with older configs.
        start_xy = start_position[:2]
        start_z = float(_env_cfg["start_z"])
    if len(goal_position) >= 3:
        goal_xy = goal_position[:2]
        goal_z = float(goal_position[2])
    else:
        # Backward compatibility with older configs.
        goal_xy = goal_position[:2]
        goal_z = float(_env_cfg["goal_z"])

    start_q = pick_collision_free_config(
        robot, start_xy, start_z, obstacles, [0.0, 0.0, start_z]
    )
    goal_q = pick_collision_free_config(
        robot, goal_xy, goal_z, obstacles, [0.0, 0.0, goal_z]
    )
    if any(
        _arm_collides_3d(
            robot,
            float(start_q[0]),
            float(start_q[1]),
            float(start_q[2]),
            obs,
        )
        for obs in obstacles
    ):
        raise ValueError(
            "Start configuration is in collision. "
            "Adjust environment.start_position/obstacles in rrp.yml."
        )
    if any(
        _arm_collides_3d(
            robot,
            float(goal_q[0]),
            float(goal_q[1]),
            float(goal_q[2]),
            obs,
        )
        for obs in obstacles
    ):
        raise ValueError(
            "Goal configuration is in collision. "
            "Adjust environment.goal_position/obstacles in rrp.yml."
        )
    logger.info(
        "Start: q=(%.3f, %.3f) z=%.3f", start_q[0], start_q[1], start_q[2]
    )
    logger.info(
        "Goal:  q=(%.3f, %.3f) z=%.3f", goal_q[0], goal_q[1], goal_q[2]
    )

    logger.info("Building 3-D C-space occupancy map …")
    grid_n = int(_planner_cfg.get("cspace_grid_n", 60))
    occ, collision_pts = build_cspace_occupancy_3d(
        robot, obstacles, clearance, grid_n=grid_n
    )

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
    logger.info("Running RRT* in 3-D joint space …")
    rrt_t0 = time.perf_counter()
    rrt_result = rrt.plan(start_q.copy(), goal_q.copy())
    rrt_elapsed = time.perf_counter() - rrt_t0
    rrt_path: list[np.ndarray] | None = (
        [np.asarray(p) for p in rrt_result] if rrt_result else None
    )
    rrt_nodes: list = list(getattr(rrt, "_nodes", []))
    rrt_len = _polyline_length(rrt_path)
    logger.info(
        "RRT*: %d waypoints, length=%.3f",
        len(rrt_path) if rrt_path else 0,
        rrt_len,
    )

    # --- SST ----------------------------------------------------------------
    sst = SSTPlanner(
        occ,
        bounds=bounds,
        max_sample_count=int(_planner_cfg["sst_max_sample_count"]),
        step_size=float(_planner_cfg["step_size"]),
        goal_tolerance=float(_planner_cfg["goal_tolerance"]),
        collision_check_count=int(_planner_cfg["collision_check_count"]),
        goal_bias=float(_planner_cfg["goal_bias"]),
        witness_radius=float(_planner_cfg["witness_radius"]),
        early_stop=True,
    )
    logger.info("Running SST in 3-D joint space …")
    sst_t0 = time.perf_counter()
    sst_result = sst.plan(start_q.copy(), goal_q.copy())
    sst_elapsed = time.perf_counter() - sst_t0
    sst_path: list[np.ndarray] | None = (
        [np.asarray(p) for p in sst_result] if sst_result else None
    )
    sst_nodes: list = list(getattr(sst, "_nodes", []))
    sst_len = _polyline_length(sst_path)
    logger.info(
        "SST: %d waypoints, length=%.3f",
        len(sst_path) if sst_path else 0,
        sst_len,
    )

    # --- Trajectory optimisation -------------------------------------------
    opt = TrajectoryOptimizer(
        occ,
        cruise_speed=float(_sim_cfg.get("race_speed", 0.6)),
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
        try:
            res = opt.optimize(rrt_path)
            rrt_traj = res.states
            rrt_traj_dur = sum(res.durations)
            rrt_traj_len = _polyline_length(res.states)
            rrt_opt_status = (
                f"{res.optimizer_status_code}: {res.optimizer_status_text}"
            )
        except Exception:
            logger.exception("RRT* TrajectoryOptimizer failed; skipping.")
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
        except Exception:
            logger.exception("SST TrajectoryOptimizer failed; skipping.")
            sst_opt_status = "exception"

    # --- Convert joint paths to Cartesian -----------------------------------
    rrt_cart = _fk_path(robot, rrt_path)
    sst_cart = _fk_path(robot, sst_path)
    rrt_traj_cart = _fk_path(robot, rrt_traj)
    sst_traj_cart = _fk_path(robot, sst_traj)

    start_cart = robot.forward_kinematics(
        float(start_q[0]), float(start_q[1]), float(start_q[2])
    )
    goal_cart = robot.forward_kinematics(
        float(goal_q[0]), float(goal_q[1]), float(goal_q[2])
    )

    # --- Figure -------------------------------------------------------------
    fig = plt.figure(figsize=(18, 6))

    x_lim = (float(_env_cfg["bounds"][0][0]), float(_env_cfg["bounds"][0][1]))
    y_lim = (float(_env_cfg["bounds"][1][0]), float(_env_cfg["bounds"][1][1]))
    z_lim = (float(_env_cfg["bounds"][2][0]), float(_env_cfg["bounds"][2][1]))

    specs = [
        (
            "RRT* — 3-D RRP SCARA",
            rrt_cart,
            rrt_len,
            "royalblue",
            rrt_traj_cart,
            {
                "steps": max(0, (len(rrt_path) - 1) if rrt_path else 0),
                "nodes": len(rrt_nodes),
                "planner_time": rrt_elapsed,
                "path_len": rrt_len,
                "traj_len": rrt_traj_len,
                "traj_dur": rrt_traj_dur,
                "path_status": "found" if rrt_path is not None else "stalled",
                "opt_status": rrt_opt_status,
            },
        ),
        (
            "SST — 3-D RRP SCARA",
            sst_cart,
            sst_len,
            "mediumseagreen",
            sst_traj_cart,
            {
                "steps": max(0, (len(sst_path) - 1) if sst_path else 0),
                "nodes": len(sst_nodes),
                "planner_time": sst_elapsed,
                "path_len": sst_len,
                "traj_len": sst_traj_len,
                "traj_dur": sst_traj_dur,
                "path_status": "found" if sst_path is not None else "stalled",
                "opt_status": sst_opt_status,
            },
        ),
    ]

    for col, (title, cart, length, color, traj_cart, metrics) in enumerate(
        specs
    ):
        ax = fig.add_subplot(1, 3, col + 1, projection="3d")

        # Obstacles
        for obs in obstacles:
            obs_color = "sienna" if (obs[3] - obs[0]) <= 0.8 else "steelblue"
            _draw_box(ax, *obs, color=obs_color, alpha=0.40)

        # Workspace annulus (ground reference)
        _workspace_annulus(
            ax,
            robot.workspace_annulus()[0],
            robot.workspace_radius(),
            robot.z_min,
            robot.z_max,
        )

        # Solution path (Cartesian EE trace)
        if cart is not None and len(cart) >= 2:
            arr = np.array(cart)
            path_alpha = 0.35 if traj_cart is not None else 1.0
            label = f"Path  {length:.2f} | {len(cart)} wpts"
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

        # Optimized trajectory
        if traj_cart is not None and len(traj_cart) >= 2:
            tarr = np.array(traj_cart)
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

        # Start / goal markers
        ax.scatter(  # type: ignore[attr-defined]
            [start_cart[0]],
            [start_cart[1]],
            [start_cart[2]],
            color="limegreen",
            s=80,
            zorder=6,
            label="Start",
        )
        ax.scatter(  # type: ignore[attr-defined]
            [goal_cart[0]],
            [goal_cart[1]],
            [goal_cart[2]],
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
        ax.legend(loc="upper right", fontsize=7)  # type: ignore[attr-defined]
        ax.view_init(elev=25, azim=-50)  # type: ignore[attr-defined]

        metrics_lines = [
            (
                f"Planner steps / nodes: "
                f"{metrics['steps']} / {metrics['nodes']}"
            ),
            f"Planner time: {_format_clock(float(metrics['planner_time']))}",
            f"Planned path length: {float(metrics['path_len']):.2f} (rad+m)",
            (
                f"Trajectory arc length: "
                f"{float(metrics['traj_len']):.2f} (rad+m)"
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

    # --- C-space scatter panel (q1-q2 plane, coloured by z) -----------------
    ax_c = fig.add_subplot(1, 3, 3)
    if collision_pts:
        cpts = np.array(collision_pts)
        sc = ax_c.scatter(
            cpts[:, 0],
            cpts[:, 1],
            c=cpts[:, 2],
            cmap="viridis",
            s=2,
            alpha=0.3,
            label="Collision configs",
        )
        plt.colorbar(sc, ax=ax_c, label="Z height (m)", fraction=0.04)

    for path, label, color in (
        (rrt_path, "RRT* path", "royalblue"),
        (sst_path, "SST path", "mediumseagreen"),
    ):
        if path is not None and len(path) >= 2:
            arr = np.array(path)
            ax_c.plot(
                arr[:, 0],
                arr[:, 1],
                color=color,
                linewidth=1.5,
                alpha=0.8,
                label=label,
            )

    ax_c.scatter(
        [float(start_q[0])],
        [float(start_q[1])],
        color="limegreen",
        s=80,
        zorder=6,
        label="Start",
    )
    ax_c.scatter(
        [float(goal_q[0])],
        [float(goal_q[1])],
        color="orangered",
        marker="*",
        s=120,
        zorder=6,
        label="Goal",
    )

    ax_c.set_xlabel("q1 (rad)")
    ax_c.set_ylabel("q2 (rad)")
    ax_c.set_title("C-space (q1–q2) — collision by height")
    ax_c.set_xlim(-math.pi, math.pi)
    ax_c.set_ylim(-math.pi, math.pi)
    ax_c.set_aspect("equal")
    ax_c.legend(loc="upper right", fontsize=7)

    plt.suptitle(
        "RRP robot (SCARA) — RRT* vs SST in 3-D cylindrical workspace",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info("Saved RRP planning example to %s", save_path)
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
