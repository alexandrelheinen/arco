"""RRP SCARA-like arm planning — RRT* vs SST in 3-D joint space.

Runs RRT* and SST planners for a two-link planar arm with a vertical
prismatic joint (RRP / SCARA-like).  The cylindrical workspace contains
two pillar obstacles (forcing XY routing) and two slab obstacles (forcing
Z routing), so a plan must combine all three joint degrees of freedom.

Figure layout (standard two-frame)
------------------------------------
* Top-left  — Workspace: both RRT* and SST FK end-effector traces,
  obstacles, start/goal markers in 3-D Cartesian space.
* Top-right — C-space (q₁, q₂, z): collision volume mesh (red), both
  joint paths/trajectories, velocity constraint ellipsoid (blue wireframe)
  centered on start.
* Bottom    — Metrics: per-planner step counts, planning times, path and
  trajectory lengths.

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
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from arco.config.palette import annotation_hex, layer_hex, obstacle_hex
from arco.kinematics import RRPRobot
from arco.planning.continuous import (
    RRTPlanner,
    SSTPlanner,
    TrajectoryOptimizer,
    TrajectoryPruner,
)
from arco.tools.logging_config import configure_logging
from arco.tools.simulator.scenes.rrp import (
    _arm_collides_3d,
    build_cspace_occupancy_3d,
    pick_collision_free_config,
)
from arco.tools.viewer.layout import StandardLayout
from arco.tools.viewer.utils import format_clock, polyline_length

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 3-D drawing helpers
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
        z_max: Top Z level for cylinder wall lines (unused — kept for API).
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


def _build_cspace_mesh(
    collision_pts: list[list[float]],
    q_bounds: tuple[float, float] = (-math.pi, math.pi),
    z_bounds: tuple[float, float] = (0.0, 4.0),
    grid_n: int = 20,
) -> np.ndarray | None:
    """Build a voxel-surface mesh from C-space collision samples.

    Args:
        collision_pts: List of ``[q1, q2, z]`` collision samples.
        q_bounds: Shared ``(q_min, q_max)`` for both revolute joints.
        z_bounds: ``(z_min, z_max)`` for the prismatic joint.
        grid_n: Visualisation grid resolution along each axis.

    Returns:
        Float array of shape ``(T, 3, 3)`` — T triangles — or ``None``
        when *collision_pts* is empty.
    """
    if not collision_pts:
        return None
    cpts = np.asarray(collision_pts, dtype=float)
    q_lo, q_hi = q_bounds
    z_lo, z_hi = z_bounds
    q_sc = (grid_n - 1) / max(q_hi - q_lo, 1e-9)
    z_sc = (grid_n - 1) / max(z_hi - z_lo, 1e-9)
    gi = np.clip(
        np.rint((cpts[:, 0] - q_lo) * q_sc).astype(int), 0, grid_n - 1
    )
    gj = np.clip(
        np.rint((cpts[:, 1] - q_lo) * q_sc).astype(int), 0, grid_n - 1
    )
    gk = np.clip(
        np.rint((cpts[:, 2] - z_lo) * z_sc).astype(int), 0, grid_n - 1
    )
    occ = np.zeros((grid_n + 2, grid_n + 2, grid_n + 2), dtype=bool)
    occ[gi + 1, gj + 1, gk + 1] = True
    inner = occ[1:-1, 1:-1, 1:-1]
    q1_ax = np.linspace(q_lo, q_hi, grid_n)
    q2_ax = np.linspace(q_lo, q_hi, grid_n)
    z_ax = np.linspace(z_lo, z_hi, grid_n)
    dq1 = q1_ax[1] - q1_ax[0] if grid_n > 1 else 1.0
    dq2 = q2_ax[1] - q2_ax[0] if grid_n > 1 else 1.0
    dz = z_ax[1] - z_ax[0] if grid_n > 1 else 1.0
    all_tris: list[np.ndarray] = []
    for di, dj, dk in (
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ):
        if di == 1:
            exp = inner & ~occ[2:, 1:-1, 1:-1]
        elif di == -1:
            exp = inner & ~occ[:-2, 1:-1, 1:-1]
        elif dj == 1:
            exp = inner & ~occ[1:-1, 2:, 1:-1]
        elif dj == -1:
            exp = inner & ~occ[1:-1, :-2, 1:-1]
        elif dk == 1:
            exp = inner & ~occ[1:-1, 1:-1, 2:]
        else:
            exp = inner & ~occ[1:-1, 1:-1, :-2]
        ii, jj, kk = np.where(exp)
        if ii.size == 0:
            continue
        q1c = q1_ax[ii]
        q2c = q2_ax[jj]
        zc = z_ax[kk]
        fc1 = q1c + di * dq1 * 0.5
        fc2 = q2c + dj * dq2 * 0.5
        fcz = zc + dk * dz * 0.5
        if di != 0:
            c00 = np.column_stack([fc1, q2c - dq2 * 0.5, zc - dz * 0.5])
            c10 = np.column_stack([fc1, q2c + dq2 * 0.5, zc - dz * 0.5])
            c11 = np.column_stack([fc1, q2c + dq2 * 0.5, zc + dz * 0.5])
            c01 = np.column_stack([fc1, q2c - dq2 * 0.5, zc + dz * 0.5])
        elif dj != 0:
            c00 = np.column_stack([q1c - dq1 * 0.5, fc2, zc - dz * 0.5])
            c10 = np.column_stack([q1c + dq1 * 0.5, fc2, zc - dz * 0.5])
            c11 = np.column_stack([q1c + dq1 * 0.5, fc2, zc + dz * 0.5])
            c01 = np.column_stack([q1c - dq1 * 0.5, fc2, zc + dz * 0.5])
        else:
            c00 = np.column_stack([q1c - dq1 * 0.5, q2c - dq2 * 0.5, fcz])
            c10 = np.column_stack([q1c + dq1 * 0.5, q2c - dq2 * 0.5, fcz])
            c11 = np.column_stack([q1c + dq1 * 0.5, q2c + dq2 * 0.5, fcz])
            c01 = np.column_stack([q1c - dq1 * 0.5, q2c + dq2 * 0.5, fcz])
        all_tris.append(np.stack([c00, c10, c11], axis=1))
        all_tris.append(np.stack([c00, c11, c01], axis=1))
    if not all_tris:
        return None
    return np.concatenate(all_tris, axis=0)


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
# Main
# ---------------------------------------------------------------------------


def main(cfg: dict, save_path: str | None = None) -> None:
    """Run RRT* and SST on the 3-D RRP arm and display the result.

    Args:
        cfg: Scenario configuration dictionary (loaded from ``rrp.yml``).
        save_path: If provided, save the figure to this path instead of
            opening an interactive window.
    """
    if save_path is not None:
        matplotlib.use("Agg")

    robot_cfg = cfg.get("robot", cfg)
    env_cfg = cfg.get("environment", cfg)
    planner_cfg = cfg.get("planner", cfg)
    sim_cfg = cfg.get("simulator", cfg)

    robot = RRPRobot(
        l1=float(robot_cfg["l1"]),
        l2=float(robot_cfg["l2"]),
        z_min=float(robot_cfg["z_min"]),
        z_max=float(robot_cfg["z_max"]),
    )
    obstacles: list[list[float]] = [
        [float(v) for v in obs] for obs in env_cfg["obstacles"]
    ]
    bounds = [tuple(b) for b in env_cfg["bounds"]]
    clearance = float(env_cfg["obstacle_clearance"])
    start_position = [float(v) for v in env_cfg["start_position"]]
    goal_position = [float(v) for v in env_cfg["goal_position"]]
    if len(start_position) >= 3:
        start_xy = start_position[:2]
        start_z = float(start_position[2])
    else:
        start_xy = start_position[:2]
        start_z = float(env_cfg["start_z"])
    if len(goal_position) >= 3:
        goal_xy = goal_position[:2]
        goal_z = float(goal_position[2])
    else:
        goal_xy = goal_position[:2]
        goal_z = float(env_cfg["goal_z"])

    start_q = pick_collision_free_config(
        robot, start_xy, start_z, obstacles, [0.0, 0.0, start_z]
    )
    goal_q = pick_collision_free_config(
        robot, goal_xy, goal_z, obstacles, [0.0, 0.0, goal_z]
    )

    for q, label in ((start_q, "Start"), (goal_q, "Goal")):
        if any(
            _arm_collides_3d(robot, float(q[0]), float(q[1]), float(q[2]), obs)
            for obs in obstacles
        ):
            raise ValueError(
                f"{label} configuration is in collision. "
                "Adjust environment.start_position/obstacles in rrp.yml."
            )

    logger.info("Building 3-D C-space occupancy map …")
    grid_n = int(planner_cfg.get("cspace_grid_n", 60))
    occ, collision_pts = build_cspace_occupancy_3d(
        robot, obstacles, clearance, grid_n=grid_n
    )

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
    logger.info("Running RRT* in 3-D joint space …")
    rrt_t0 = time.perf_counter()
    rrt_result = rrt.plan(start_q.copy(), goal_q.copy())
    rrt_elapsed = time.perf_counter() - rrt_t0
    rrt_path: list[np.ndarray] | None = (
        [np.asarray(p) for p in rrt_result] if rrt_result else None
    )
    rrt_nodes: list = list(getattr(rrt, "_nodes", []))
    rrt_len = polyline_length(rrt_path)

    # --- SST ----------------------------------------------------------------
    sst = SSTPlanner(
        occ,
        bounds=bounds,
        max_sample_count=int(planner_cfg["sst_max_sample_count"]),
        step_size=planner_cfg["step_size"],
        goal_tolerance=float(planner_cfg["goal_tolerance"]),
        collision_check_count=int(planner_cfg["collision_check_count"]),
        goal_bias=float(planner_cfg["goal_bias"]),
        witness_radius=float(planner_cfg["witness_radius"]),
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
    sst_len = polyline_length(sst_path)

    # --- Trajectory optimisation -------------------------------------------
    _enable_pruning = bool(planner_cfg.get("enable_pruning", False))
    pruner = (
        TrajectoryPruner(
            occ,
            step_size=np.asarray(planner_cfg["step_size"], dtype=float),
            collision_check_count=int(planner_cfg["collision_check_count"]),
        )
        if _enable_pruning
        else None
    )
    opt = TrajectoryOptimizer(
        occ,
        cruise_speed=float(sim_cfg.get("race_speed", 0.6)),
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
    rrt_opt_status = "not-run"
    sst_opt_status = "not-run"

    for path, _traj_ref, _dur_ref, _opt_ref, _label, _durs_ref in (
        (
            rrt_path,
            None,
            0.0,
            "rrt",
            "RRT*",
            [],
        ),
    ):
        pass  # parsed below, kept for structure

    if rrt_path is not None:
        rrt_path = (
            pruner.prune(rrt_path) if pruner is not None else list(rrt_path)
        )
        try:
            res = opt.optimize(rrt_path)
            rrt_traj = res.states
            rrt_traj_dur = sum(res.durations) if res.durations else 0.0
            rrt_opt_status = (
                f"{res.optimizer_status_code}: {res.optimizer_status_text}"
            )
        except Exception:
            logger.exception("RRT* TrajectoryOptimizer failed; skipping.")
            rrt_opt_status = "exception"
    if sst_path is not None:
        sst_path = (
            pruner.prune(sst_path) if pruner is not None else list(sst_path)
        )
        try:
            res = opt.optimize(sst_path)
            sst_traj = res.states
            sst_traj_dur = sum(res.durations) if res.durations else 0.0
            sst_opt_status = (
                f"{res.optimizer_status_code}: {res.optimizer_status_text}"
            )
        except Exception:
            logger.exception("SST TrajectoryOptimizer failed; skipping.")
            sst_opt_status = "exception"

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
    fig, ax_ws, ax_cs, ax_bottom = StandardLayout.create(
        title="RRP robot (SCARA) — RRT* vs SST in 3-D cylindrical workspace",
        ws_3d=True,
        cs_3d=True,
    )

    x_lim = (float(env_cfg["bounds"][0][0]), float(env_cfg["bounds"][0][1]))
    y_lim = (float(env_cfg["bounds"][1][0]), float(env_cfg["bounds"][1][1]))
    z_lim = (float(env_cfg["bounds"][2][0]), float(env_cfg["bounds"][2][1]))
    q_bnd = (-math.pi, math.pi)
    z_bnd = (robot.z_min, robot.z_max)

    # ---- ax_ws: Combined 3-D Cartesian workspace ---------------------------
    for obs in obstacles:
        _draw_box(ax_ws, *obs, color=obstacle_hex(), alpha=0.40)
    _workspace_annulus(
        ax_ws,
        robot.workspace_annulus()[0],
        robot.workspace_radius(),
        robot.z_min,
        robot.z_max,
    )

    for cart, lbl, path_key in (
        (rrt_cart, "RRT* path", "rrt"),
        (sst_cart, "SST path", "sst"),
    ):
        if cart is not None and len(cart) >= 2:
            arr = np.array(cart)
            ax_ws.plot(
                arr[:, 0],
                arr[:, 1],
                arr[:, 2],  # type: ignore[attr-defined]
                color=layer_hex(path_key, "path"),
                linewidth=1.5,
                alpha=0.7,
                label=lbl,
            )
    for cart, lbl, traj_key in (
        (rrt_traj_cart, "RRT* traj", "rrt"),
        (sst_traj_cart, "SST traj", "sst"),
    ):
        if cart is not None and len(cart) >= 2:
            tarr = np.array(cart)
            ax_ws.plot(
                tarr[:, 0],
                tarr[:, 1],
                tarr[:, 2],  # type: ignore[attr-defined]
                "o-",
                color=layer_hex(traj_key, "trajectory"),
                linewidth=2.5,
                markersize=3,
                alpha=0.9,
                label=lbl,
            )

    ax_ws.scatter(
        [start_cart[0]],
        [start_cart[1]],
        [start_cart[2]],  # type: ignore[attr-defined]
        color=annotation_hex(),
        s=80,
        zorder=6,
        label="Start",
    )
    ax_ws.scatter(
        [goal_cart[0]],
        [goal_cart[1]],
        [goal_cart[2]],  # type: ignore[attr-defined]
        color=annotation_hex(),
        marker="x",
        linewidths=2,
        s=80,
        zorder=6,
        label="Goal",
    )
    ax_ws.set_xlim(*x_lim)  # type: ignore[attr-defined]
    ax_ws.set_ylim(*y_lim)  # type: ignore[attr-defined]
    ax_ws.set_zlim(*z_lim)  # type: ignore[attr-defined]
    ax_ws.set_xlabel("X (m)")  # type: ignore[attr-defined]
    ax_ws.set_ylabel("Y (m)")  # type: ignore[attr-defined]
    ax_ws.set_zlabel("Z (m)")  # type: ignore[attr-defined]
    ax_ws.set_title("Workspace (Cartesian)")  # type: ignore[attr-defined]
    ax_ws.set_box_aspect(  # type: ignore[attr-defined]
        [x_lim[1] - x_lim[0], y_lim[1] - y_lim[0], z_lim[1] - z_lim[0]]
    )
    ax_ws.legend(loc="upper right", fontsize=7)  # type: ignore[attr-defined]
    ax_ws.view_init(elev=25, azim=-50)  # type: ignore[attr-defined]

    # ---- ax_cs: C-space (q₁, q₂, z) with collision mesh -------------------
    if collision_pts:
        tris = _build_cspace_mesh(
            collision_pts, q_bounds=q_bnd, z_bounds=z_bnd
        )
        if tris is not None and len(tris) > 0:
            ax_cs.add_collection3d(  # type: ignore[attr-defined]
                Poly3DCollection(
                    list(tris),
                    alpha=0.25,
                    facecolor=obstacle_hex(),
                    edgecolor="none",
                    linewidth=0,
                )
            )
        else:
            cpts_arr = np.array(collision_pts)
            ax_cs.scatter(  # type: ignore[attr-defined]
                cpts_arr[:, 0],
                cpts_arr[:, 1],
                cpts_arr[:, 2],
                c=obstacle_hex(),
                s=2,
                alpha=0.30,
            )

    for path, lbl, color in (
        (rrt_path, "RRT* path", layer_hex("rrt", "path")),
        (sst_path, "SST path", layer_hex("sst", "path")),
    ):
        if path is not None and len(path) >= 2:
            arr = np.array(path)
            ax_cs.plot(
                arr[:, 0],
                arr[:, 1],
                arr[:, 2],  # type: ignore[attr-defined]
                color=color,
                linewidth=1.5,
                alpha=0.85,
                label=lbl,
            )
    for traj, lbl, key in (
        (rrt_traj, "RRT* traj", "rrt"),
        (sst_traj, "SST traj", "sst"),
    ):
        if traj is not None and len(traj) >= 2:
            tarr = np.array(traj)
            ax_cs.plot(
                tarr[:, 0],
                tarr[:, 1],
                tarr[:, 2],  # type: ignore[attr-defined]
                "o-",
                color=layer_hex(key, "trajectory"),
                linewidth=2.0,
                markersize=3,
                alpha=0.9,
                label=lbl,
            )

    max_ang_vel = float(sim_cfg.get("max_ang_vel", 1.5))
    max_lin_vel = float(sim_cfg.get("max_lin_vel", 1.0))
    u = np.linspace(0, 2 * math.pi, 24)
    v = np.linspace(0, math.pi, 12)
    ex = max_ang_vel * np.outer(np.cos(u), np.sin(v)) + float(start_q[0])
    ey = max_ang_vel * np.outer(np.sin(u), np.sin(v)) + float(start_q[1])
    ez = max_lin_vel * np.outer(np.ones_like(u), np.cos(v)) + float(start_q[2])
    ax_cs.plot_wireframe(ex, ey, ez, color="blue", alpha=0.15, linewidth=0.5)  # type: ignore[attr-defined]

    ax_cs.scatter(
        [float(start_q[0])],
        [float(start_q[1])],
        [float(start_q[2])],  # type: ignore[attr-defined]
        color=annotation_hex(),
        s=80,
        zorder=6,
        label="Start",
    )
    ax_cs.scatter(
        [float(goal_q[0])],
        [float(goal_q[1])],
        [float(goal_q[2])],  # type: ignore[attr-defined]
        color=annotation_hex(),
        marker="x",
        linewidths=2,
        s=80,
        zorder=6,
        label="Goal",
    )
    ax_cs.set_xlabel("q₁ (rad)")  # type: ignore[attr-defined]
    ax_cs.set_ylabel("q₂ (rad)")  # type: ignore[attr-defined]
    ax_cs.set_zlabel("z (m)")  # type: ignore[attr-defined]
    ax_cs.set_title("C-space (q₁, q₂, z)")  # type: ignore[attr-defined]
    ax_cs.set_xlim(-math.pi, math.pi)  # type: ignore[attr-defined]
    ax_cs.set_ylim(-math.pi, math.pi)  # type: ignore[attr-defined]
    ax_cs.set_zlim(*z_bnd)  # type: ignore[attr-defined]
    ax_cs.set_box_aspect(  # type: ignore[attr-defined]
        [2 * math.pi, 2 * math.pi, z_bnd[1] - z_bnd[0]]
    )
    ax_cs.view_init(elev=20, azim=-55)  # type: ignore[attr-defined]
    ax_cs.legend(loc="upper right", fontsize=7)  # type: ignore[attr-defined]

    # ---- Bottom: metrics ---------------------------------------------------
    StandardLayout.write_metrics(
        ax_bottom,
        [
            f"RRT*  steps/nodes: "
            f"{max(0, len(rrt_path)-1 if rrt_path else 0)}/{len(rrt_nodes)} | "
            f"time: {format_clock(rrt_elapsed)} | "
            f"path: {rrt_len:.2f} | "
            f"traj: {format_clock(rrt_traj_dur)} | {rrt_opt_status}",
            f"SST   steps/nodes: "
            f"{max(0, len(sst_path)-1 if sst_path else 0)}/{len(sst_nodes)} | "
            f"time: {format_clock(sst_elapsed)} | "
            f"path: {sst_len:.2f} | "
            f"traj: {format_clock(sst_traj_dur)} | {sst_opt_status}",
        ],
    )

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info("Saved RRP planning example to %s", save_path)
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
