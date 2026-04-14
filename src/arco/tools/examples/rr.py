"""2-D RR robot arm planning — RRT* vs SST in joint space.

Runs RRT* and SST planners side-by-side for a two-link planar arm
navigating from a start to a goal configuration while avoiding a
rectangular obstacle.  Planning is done in joint space (theta1, theta2);
resulting paths are converted to Cartesian space via forward kinematics
for visualization.

Figure layout (3 subplots in a row)
------------------------------------
* Left   — Combined Cartesian workspace: both RRT* and SST end-effector
  traces, obstacles, robot at start/goal.
* Middle — Joint C-space (θ₁, θ₂): collision configs (red scatter), both
  joint paths/trajectories, velocity constraint disc (blue circle).
* Right  — Lyapunov function V(t) = ‖q(t) − q_goal‖ for both trajectories
  with a sliding-window highlight of the last T/10 seconds.

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

from arco.config.palette import annotation_hex, layer_hex, obstacle_hex
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

logger = logging.getLogger(__name__)

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


def main(cfg: dict, save_path: str | None = None) -> None:
    """Run the RR planning example and display or save the figure.

    Args:
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

    # --- Setup -----------------------------------------------------------
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

    # --- Occupancy map ---------------------------------------------------
    logger.info("Building C-space occupancy map …")
    occ, collision_pts = build_cspace_occupancy(
        robot, obstacles, bounds, clearance
    )

    # --- RRT* ------------------------------------------------------------
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
    sst_len = _polyline_length(sst_path)
    logger.info(
        "SST: %d waypoints, length=%.3f rad",
        len(sst_path) if sst_path else 0,
        sst_len,
    )

    # --- Trajectory optimization -----------------------------------------
    pruner = TrajectoryPruner(
        occ,
        collision_check_count=int(planner_cfg["collision_check_count"]),
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
        path: list[np.ndarray] | None,
        label: str,
    ) -> tuple[list[np.ndarray] | None, float, float, str, list[float]]:
        if path is None or len(path) < 2:
            return None, 0.0, 0.0, "no-path", []
        path = pruner.prune(path)
        try:
            result = optimizer.optimize(path)
            traj = list(result.states)
            durs: list[float] = list(result.durations) if result.durations else []
            dur = float(sum(durs)) if durs else 0.0
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
            durs = []
            dur = 0.0
            status = "error"
        return traj, _polyline_length(traj), dur, status, durs

    rrt_traj, rrt_traj_len, rrt_traj_dur, rrt_opt_status, rrt_durs = _optimize(
        rrt_path, "RRT*"
    )
    sst_traj, sst_traj_len, sst_traj_dur, sst_opt_status, sst_durs = _optimize(
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
            facecolor=obstacle_hex(),
            alpha=0.5,
            label="Obstacle",
        )

    rrt_cart = _fk_path(robot, rrt_path)
    sst_cart = _fk_path(robot, sst_path)
    rrt_traj_cart = _fk_path(robot, rrt_traj)
    sst_traj_cart = _fk_path(robot, sst_traj)

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

    # --- Figure ----------------------------------------------------------
    fig, (ax_ws, ax_cs, ax_lv) = plt.subplots(1, 3, figsize=(18, 6))

    # ---- ax_ws: Combined Cartesian workspace ----------------------------
    ax_ws.fill(outer_x, outer_y, alpha=0.07, color="steelblue")
    if r_min > _INNER_RADIUS_THRESHOLD:
        ax_ws.fill(inner_x, inner_y, alpha=0.25, color="white")
    ax_ws.plot(outer_x, outer_y, color="steelblue", linewidth=0.8, alpha=0.5)
    if r_min > _INNER_RADIUS_THRESHOLD:
        ax_ws.plot(inner_x, inner_y, color="steelblue", linewidth=0.8, alpha=0.5)

    first_obs = True
    for obs in obstacles:
        patch = _obs_patch(obs)
        if not first_obs:
            patch.set_label("_nolegend_")
        ax_ws.add_patch(patch)
        first_obs = False

    _draw_arm(ax_ws, robot, float(start_q[0]), float(start_q[1]),
              color=annotation_hex(), alpha=0.8, label="Start arm")
    _draw_arm(ax_ws, robot, float(goal_q[0]), float(goal_q[1]),
              color=annotation_hex(), alpha=0.5, label="Goal arm")

    if rrt_cart is not None and len(rrt_cart) >= 2:
        arr = np.array(rrt_cart)
        ax_ws.plot(arr[:, 0], arr[:, 1],
                   color=layer_hex("rrt", "path"), linewidth=1.5, alpha=0.7,
                   label="RRT* path")
    if rrt_traj_cart is not None and len(rrt_traj_cart) >= 2:
        tarr = np.array(rrt_traj_cart)
        ax_ws.plot(tarr[:, 0], tarr[:, 1], "o-",
                   color=layer_hex("rrt", "trajectory"), linewidth=2.0,
                   markersize=3, alpha=0.9, label="RRT* traj")
    if sst_cart is not None and len(sst_cart) >= 2:
        arr = np.array(sst_cart)
        ax_ws.plot(arr[:, 0], arr[:, 1],
                   color=layer_hex("sst", "path"), linewidth=1.5, alpha=0.7,
                   label="SST path")
    if sst_traj_cart is not None and len(sst_traj_cart) >= 2:
        tarr = np.array(sst_traj_cart)
        ax_ws.plot(tarr[:, 0], tarr[:, 1], "o-",
                   color=layer_hex("sst", "trajectory"), linewidth=2.0,
                   markersize=3, alpha=0.9, label="SST traj")

    sx, sy = robot.forward_kinematics(float(start_q[0]), float(start_q[1]))
    gx, gy = robot.forward_kinematics(float(goal_q[0]), float(goal_q[1]))
    ax_ws.plot(sx, sy, "s", color=annotation_hex(), ms=8, zorder=9)
    ax_ws.plot(gx, gy, "x", color=annotation_hex(), ms=8, mew=2, zorder=9)

    ax_ws.set_aspect("equal")
    ax_ws.set_xlabel("X (m)")
    ax_ws.set_ylabel("Y (m)")
    ax_ws.set_title("RR robot — Cartesian workspace")
    ax_ws.legend(loc="upper left", fontsize=7)
    ax_ws.grid(True, alpha=0.3)

    metrics_lines = [
        f"RRT* steps/nodes: {max(0, len(rrt_path)-1 if rrt_path else 0)}/{len(rrt_nodes)}",
        f"RRT* time: {_format_clock(rrt_elapsed)} | len: {rrt_len:.2f} rad",
        f"RRT* traj dur: {_format_clock(rrt_traj_dur)} | {rrt_opt_status}",
        f"SST steps/nodes: {max(0, len(sst_path)-1 if sst_path else 0)}/{len(sst_nodes)}",
        f"SST time: {_format_clock(sst_elapsed)} | len: {sst_len:.2f} rad",
        f"SST traj dur: {_format_clock(sst_traj_dur)} | {sst_opt_status}",
    ]
    ax_ws.text(
        0.02, 0.98, "\n".join(metrics_lines),
        transform=ax_ws.transAxes, va="top", ha="left", fontsize=7,
        color="black",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.80,
              "edgecolor": "none"},
    )

    # ---- ax_cs: C-space (θ₁, θ₂) with velocity constraint disc ----------
    if collision_pts:
        cpts = np.array(collision_pts)
        ax_cs.scatter(cpts[:, 0], cpts[:, 1], s=1, c=obstacle_hex(),
                      alpha=0.3, label="C-space obstacle")

    if rrt_path is not None:
        arr = np.array(rrt_path)
        ax_cs.plot(arr[:, 0], arr[:, 1],
                   color=layer_hex("rrt", "path"), linewidth=1.5, alpha=0.8,
                   label="RRT* path")
    if rrt_traj is not None:
        tarr = np.array(rrt_traj)
        ax_cs.plot(tarr[:, 0], tarr[:, 1], "o-",
                   color=layer_hex("rrt", "trajectory"), linewidth=2.0,
                   markersize=3, alpha=0.9, label="RRT* traj")
    if sst_path is not None:
        arr = np.array(sst_path)
        ax_cs.plot(arr[:, 0], arr[:, 1],
                   color=layer_hex("sst", "path"), linewidth=1.5, alpha=0.8,
                   label="SST path")
    if sst_traj is not None:
        tarr = np.array(sst_traj)
        ax_cs.plot(tarr[:, 0], tarr[:, 1], "o-",
                   color=layer_hex("sst", "trajectory"), linewidth=2.0,
                   markersize=3, alpha=0.9, label="SST traj")

    # Velocity constraint disc (blue circle) around start and goal
    cruise_speed = float(sim_cfg.get("race_speed", 1.0))
    for center, marker in ((start_q, "s"), (goal_q, "x")):
        circ = mpatches.Circle(
            (float(center[0]), float(center[1])), radius=cruise_speed,
            linewidth=1.2, edgecolor="blue", facecolor="none", alpha=0.5,
            linestyle="--",
        )
        ax_cs.add_patch(circ)
    ax_cs.plot(float(start_q[0]), float(start_q[1]), "s",
               color=annotation_hex(), ms=10, zorder=9, label="Start")
    ax_cs.plot(float(goal_q[0]), float(goal_q[1]), "x",
               color=annotation_hex(), ms=8, mew=2, zorder=9, label="Goal")
    ax_cs.set_xlim(*bounds[0])
    ax_cs.set_ylim(*bounds[1])
    ax_cs.set_xlabel("θ₁ (rad)")
    ax_cs.set_ylabel("θ₂ (rad)")
    ax_cs.set_title("C-space (θ₁, θ₂)")
    ax_cs.legend(loc="upper right", fontsize=7)
    ax_cs.grid(True, alpha=0.3)
    ax_cs.set_aspect("equal")

    # ---- ax_lv: Lyapunov function V(t) = ‖q(t) − goal_q‖ ---------------
    goal_arr = np.asarray(goal_q)
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
    ax_lv.set_ylabel("V(t) = ‖q − goal‖ (rad)")
    ax_lv.set_title("Lyapunov function")
    ax_lv.legend(loc="upper right", fontsize=7)
    ax_lv.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        logger.info("Saved RR planning example to %s", save_path)
    else:
        plt.show()


if __name__ == "__main__":
    import yaml as _yaml

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
