"""2-D RR robot arm planning — RRT* vs SST in joint space.

Runs RRT* and SST planners for a two-link planar arm navigating from a
start to a goal configuration while avoiding a rectangular obstacle.
Planning is done in joint space (theta1, theta2); results are converted
to Cartesian space via forward kinematics for workspace rendering.

Figure layout (standard two-frame)
------------------------------------
* Top-left  — Workspace: both RRT* and SST end-effector traces, obstacles,
  robot at start/goal, annotated with both planners' results overlaid.
* Top-right — C-space (θ₁, θ₂): collision configs (red scatter), both
  joint paths/trajectories, velocity constraint disc (blue circle).
* Bottom    — Metrics: per-planner step counts, planning times, path and
  trajectory lengths.

Usage
-----
Run interactively (opens a matplotlib window)::

    python tools/examples/rr.py

Save the output image without opening a window::

    python tools/examples/rr.py --save path/to/output.png
"""

from __future__ import annotations

import argparse
import logging
import math
import time

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from arco.config.palette import annotation_hex, obstacle_hex
from arco.kinematics import RRRobot
from arco.planning.continuous import (
    RRTPlanner,
    SSTPlanner,
    TrajectoryOptimizer,
    TrajectoryPruner,
)
from arco.tools.logging_config import configure_logging
from arco.tools.simulator.scenes.rr import (
    build_cspace_occupancy,
    pick_collision_free_ik,
)
from arco.tools.viewer import FrameRenderer, SceneSnapshot
from arco.tools.viewer.layout import StandardLayout
from arco.tools.viewer.utils import format_clock, polyline_length

logger = logging.getLogger(__name__)

# Minimum annulus inner radius below which the inner-hole outline is skipped.
_INNER_RADIUS_THRESHOLD: float = 1e-6


# ---------------------------------------------------------------------------
# Snapshot builders
# ---------------------------------------------------------------------------


def _build_rr_cs_snapshot(
    planner: str,
    collision_pts: list[list[float]],
    start_q: np.ndarray,
    goal_q: np.ndarray,
    path: list[np.ndarray] | None,
    traj: list[np.ndarray] | None,
    *,
    include_obstacles: bool = True,
) -> SceneSnapshot:
    """Build a C-space SceneSnapshot (θ₁, θ₂) for the RR example.

    Args:
        planner: Planner key, e.g. ``"rrt"`` or ``"sst"``.
        collision_pts: Sampled joint-space collision configurations.
        start_q: Start joint config ``[θ₁, θ₂]``.
        goal_q: Goal joint config ``[θ₁, θ₂]``.
        path: Raw joint-space path.
        traj: Optimised joint-space trajectory.
        include_obstacles: When ``False`` the obstacle list is left empty.

    Returns:
        A :class:`~arco.tools.viewer.SceneSnapshot` in joint space.
    """
    obs: list[list[float]] = (
        [[float(p[0]), float(p[1])] for p in collision_pts]
        if include_obstacles
        else []
    )
    return SceneSnapshot.from_planning_result(
        scenario="rr",
        planner=planner,
        start=[float(start_q[0]), float(start_q[1])],
        goal=[float(goal_q[0]), float(goal_q[1])],
        obstacles=obs,
        found_path=(
            [[float(p[0]), float(p[1])] for p in path] if path else None
        ),
        adjusted_trajectory=(
            [[float(p[0]), float(p[1])] for p in traj] if traj else None
        ),
    )


def _build_rr_ws_snapshot(
    planner: str,
    robot: RRRobot,
    start_q: np.ndarray,
    goal_q: np.ndarray,
    path: list[np.ndarray] | None,
    traj: list[np.ndarray] | None,
) -> SceneSnapshot:
    """Build a workspace SceneSnapshot (FK-transformed x, y) for the RR example.

    Args:
        planner: Planner key, e.g. ``"rrt"`` or ``"sst"``.
        robot: The :class:`~arco.kinematics.RRRobot` instance.
        start_q: Start joint config ``[θ₁, θ₂]``.
        goal_q: Goal joint config ``[θ₁, θ₂]``.
        path: Raw joint-space path.
        traj: Optimised joint-space trajectory.

    Returns:
        A :class:`~arco.tools.viewer.SceneSnapshot` in Cartesian workspace.
    """
    start_xy = robot.forward_kinematics(float(start_q[0]), float(start_q[1]))
    goal_xy = robot.forward_kinematics(float(goal_q[0]), float(goal_q[1]))
    return SceneSnapshot.from_planning_result(
        scenario="rr",
        planner=planner,
        start=list(start_xy),
        goal=list(goal_xy),
        obstacles=[],  # drawn as AABB patches manually
        found_path=(
            [
                list(robot.forward_kinematics(float(p[0]), float(p[1])))
                for p in path
            ]
            if path
            else None
        ),
        adjusted_trajectory=(
            [
                list(robot.forward_kinematics(float(p[0]), float(p[1])))
                for p in traj
            ]
            if traj
            else None
        ),
    )


# ---------------------------------------------------------------------------
# Drawing helpers
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
# Main
# ---------------------------------------------------------------------------


def main(cfg: dict, save_path: str | None = None) -> None:
    """Run the RR planning example and display or save the figure.

    Args:
        cfg: Scenario configuration dictionary (loaded from ``rr.yml``).
        save_path: Optional path to save the output image.  When ``None``
            an interactive matplotlib window is opened instead.
    """
    configure_logging()
    if save_path is None:
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
        save_path = args.save or None

    if save_path is not None:
        matplotlib.use("Agg")

    robot_cfg = cfg.get("robot", cfg)
    env_cfg = cfg.get("environment", cfg)
    planner_cfg = cfg.get("planner", cfg)
    sim_cfg = cfg.get("simulator", cfg)

    robot = RRRobot(l1=float(robot_cfg["l1"]), l2=float(robot_cfg["l2"]))
    obstacles: list[list[float]] = [
        [float(v) for v in obs] for obs in env_cfg["obstacles"]
    ]
    bounds = [tuple(b) for b in env_cfg["bounds"]]
    clearance = float(env_cfg["obstacle_clearance"])
    start_position = [float(v) for v in env_cfg["start_position"]]
    goal_position = [float(v) for v in env_cfg["goal_position"]]

    start_q = pick_collision_free_ik(
        robot, start_position, obstacles, [-2.2, 1.8]
    )
    goal_q = pick_collision_free_ik(
        robot, goal_position, obstacles, [1.0, -1.6]
    )

    logger.info("Start joint config: (%.3f, %.3f)", start_q[0], start_q[1])
    logger.info("Goal  joint config: (%.3f, %.3f)", goal_q[0], goal_q[1])

    logger.info("Building C-space occupancy map …")
    occ, collision_pts = build_cspace_occupancy(
        robot, obstacles, bounds, clearance
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
    logger.info("Running RRT* in joint space …")
    rrt_t0 = time.perf_counter()
    rrt_result = rrt.plan(start_q, goal_q)
    rrt_elapsed = time.perf_counter() - rrt_t0
    rrt_path: list[np.ndarray] | None = (
        [np.asarray(p) for p in rrt_result] if rrt_result else None
    )
    rrt_feasible = rrt_path is not None
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
    logger.info("Running SST in joint space …")
    sst_t0 = time.perf_counter()
    sst_result = sst.plan(start_q, goal_q)
    sst_elapsed = time.perf_counter() - sst_t0
    sst_path: list[np.ndarray] | None = (
        [np.asarray(p) for p in sst_result] if sst_result else None
    )
    sst_feasible = sst_path is not None
    sst_nodes: list = list(getattr(sst, "_nodes", []))
    sst_len = polyline_length(sst_path)

    # --- Trajectory optimisation --------------------------------------------
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
    optimizer = TrajectoryOptimizer(
        occ,
        cruise_speed=float(sim_cfg.get("race_speed", 1.0)),
        weight_time=10.0,
        weight_deviation=1.0,
        weight_velocity=1.0,
        weight_collision=5.0,
        sample_count=1,
        max_iter=200,
    )

    def _optimize(
        path: list[np.ndarray] | None, label: str
    ) -> tuple[list[np.ndarray] | None, float, str]:
        if path is None or len(path) < 2:
            return None, 0.0, "no-path"
        path = pruner.prune(path) if pruner is not None else list(path)
        try:
            result = optimizer.optimize(path)
            traj = list(result.states)
            durs: list[float] = (
                list(result.durations) if result.durations else []
            )
            dur = float(sum(durs))
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
        return traj, dur, status

    rrt_traj, rrt_traj_dur, rrt_opt_status = _optimize(rrt_path, "RRT*")
    sst_traj, sst_traj_dur, sst_opt_status = _optimize(sst_path, "SST")

    # --- Workspace geometry -------------------------------------------------
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
            facecolor=obstacle_hex(),
            alpha=0.5,
            label="Obstacle",
        )

    rrt_cs_snap = _build_rr_cs_snapshot(
        "rrt",
        collision_pts,
        start_q,
        goal_q,
        rrt_path,
        rrt_traj,
        include_obstacles=True,
    )
    sst_cs_snap = _build_rr_cs_snapshot(
        "sst",
        collision_pts,
        start_q,
        goal_q,
        sst_path,
        sst_traj,
        include_obstacles=False,
    )
    rrt_ws_snap = _build_rr_ws_snapshot(
        "rrt", robot, start_q, goal_q, rrt_path, rrt_traj
    )
    sst_ws_snap = _build_rr_ws_snapshot(
        "sst", robot, start_q, goal_q, sst_path, sst_traj
    )

    rrt_cart = _fk_path(robot, rrt_path)
    sst_cart = _fk_path(robot, sst_path)
    rrt_traj_cart = _fk_path(robot, rrt_traj)
    sst_traj_cart = _fk_path(robot, sst_traj)

    # --- Figure -------------------------------------------------------------
    fig, ax_ws, ax_cs, ax_bottom = StandardLayout.create(
        title="RR robot (2-D) — RRT* vs SST in joint space"
    )

    # ---- ax_ws: Combined Cartesian workspace --------------------------------
    ax_ws.fill(outer_x, outer_y, alpha=0.07, color="steelblue")
    if r_min > _INNER_RADIUS_THRESHOLD:
        ax_ws.fill(inner_x, inner_y, alpha=0.25, color="white")
    ax_ws.plot(outer_x, outer_y, color="steelblue", linewidth=0.8, alpha=0.5)
    if r_min > _INNER_RADIUS_THRESHOLD:
        ax_ws.plot(
            inner_x, inner_y, color="steelblue", linewidth=0.8, alpha=0.5
        )

    first_obs = True
    for obs in obstacles:
        patch = _obs_patch(obs)
        if not first_obs:
            patch.set_label("_nolegend_")
        ax_ws.add_patch(patch)
        first_obs = False

    _draw_arm(
        ax_ws,
        robot,
        float(start_q[0]),
        float(start_q[1]),
        color=annotation_hex(),
        alpha=0.8,
        label="Start arm",
    )
    _draw_arm(
        ax_ws,
        robot,
        float(goal_q[0]),
        float(goal_q[1]),
        color=annotation_hex(),
        alpha=0.5,
        label="Goal arm",
    )

    FrameRenderer(draw_tree=False, draw_obstacles=False).render(
        ax_ws, rrt_ws_snap
    )
    FrameRenderer(
        draw_tree=False, draw_obstacles=False, draw_start_goal=False
    ).render(ax_ws, sst_ws_snap)

    ax_ws.set_aspect("equal")
    ax_ws.set_xlabel("X (m)")
    ax_ws.set_ylabel("Y (m)")
    ax_ws.set_title("Workspace (Cartesian)")
    ax_ws.legend(loc="upper left", fontsize=7)
    ax_ws.grid(True, alpha=0.3)

    # ---- ax_cs: C-space (θ₁, θ₂) with velocity constraint disc ------------
    FrameRenderer(draw_tree=False).render(ax_cs, rrt_cs_snap)
    FrameRenderer(
        draw_tree=False, draw_obstacles=False, draw_start_goal=False
    ).render(ax_cs, sst_cs_snap)

    cruise_speed = float(sim_cfg.get("race_speed", 1.0))
    for center in (start_q, goal_q):
        circ = mpatches.Circle(
            (float(center[0]), float(center[1])),
            radius=cruise_speed,
            linewidth=1.2,
            edgecolor="blue",
            facecolor="none",
            alpha=0.5,
            linestyle="--",
        )
        ax_cs.add_patch(circ)
    ax_cs.set_xlim(*bounds[0])
    ax_cs.set_ylim(*bounds[1])
    ax_cs.set_xlabel("θ₁ (rad)")
    ax_cs.set_ylabel("θ₂ (rad)")
    ax_cs.set_title("C-space (θ₁, θ₂)")
    ax_cs.legend(loc="upper right", fontsize=7)
    ax_cs.grid(True, alpha=0.3)
    ax_cs.set_aspect("equal")

    # ---- Bottom: metrics ---------------------------------------------------
    StandardLayout.write_metrics(
        ax_bottom,
        [
            f"RRT*  steps/nodes: "
            f"{max(0, len(rrt_path)-1 if rrt_path else 0)}/{len(rrt_nodes)} | "
            f"time: {format_clock(rrt_elapsed)} | "
            f"path: {rrt_len:.2f} rad | "
            f"traj: {format_clock(rrt_traj_dur)} | {rrt_opt_status}",
            f"SST   steps/nodes: "
            f"{max(0, len(sst_path)-1 if sst_path else 0)}/{len(sst_nodes)} | "
            f"time: {format_clock(sst_elapsed)} | "
            f"path: {sst_len:.2f} rad | "
            f"traj: {format_clock(sst_traj_dur)} | {sst_opt_status}",
        ],
    )

    plt.tight_layout()

    if save_path is not None:
        import os

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved RR planning example to %s", save_path)
    else:
        plt.show()


if __name__ == "__main__":
    import yaml as _yaml

    configure_logging()
    parser = argparse.ArgumentParser(
        description="RR robot arm planning example"
    )
    parser.add_argument(
        "scenario", metavar="FILE", help="Path to scenario YAML file."
    )
    parser.add_argument(
        "--save",
        metavar="PATH",
        default="",
        help="Save the figure to this path instead of showing it.",
    )
    args = parser.parse_args()
    with open(args.scenario) as _fh:
        _cfg = _yaml.safe_load(_fh) or {}
    main(_cfg, save_path=args.save or None)
