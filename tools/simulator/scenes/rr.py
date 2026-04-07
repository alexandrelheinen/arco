"""RR 2-D robot arm dual-planner scene.

:class:`RRScene` builds a C-space occupancy map for a two-link planar arm,
then runs both RRT* and SST planners between a start and goal joint
configuration.

The arm navigates around a rectangular Cartesian obstacle.  Planning is
performed in joint space (theta1, theta2); resulting paths are converted to
Cartesian end-effector traces via forward kinematics for visualization.

Module-level helpers :func:`_segment_intersects_rect`, :func:`_arm_collides`,
and :func:`build_cspace_occupancy` are exported so that
``tools/examples/rr_planning.py`` can import them without duplication.
"""

from __future__ import annotations

import copy
import math
import time
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Shared collision / geometry helpers (exported)
# ---------------------------------------------------------------------------


def _polyline_length(path: list[np.ndarray] | None) -> float:
    """Return total Euclidean arc length for a waypoint sequence.

    Args:
        path: Ordered list of numpy arrays, or ``None``.

    Returns:
        Total arc length in the same units as the waypoints.
    """
    if path is None or len(path) < 2:
        return 0.0
    return sum(
        float(np.linalg.norm(path[i + 1] - path[i]))
        for i in range(len(path) - 1)
    )


def _segment_intersects_rect(
    p1: tuple[float, float],
    p2: tuple[float, float],
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> bool:
    """Cohen-Sutherland parametric clipping of a segment against an AABB.

    Also returns ``True`` if either endpoint is inside the rectangle.

    Args:
        p1: Start point of segment ``(x, y)``.
        p2: End point of segment ``(x, y)``.
        xmin: Left bound of rectangle.
        ymin: Bottom bound of rectangle.
        xmax: Right bound of rectangle.
        ymax: Top bound of rectangle.

    Returns:
        ``True`` if the segment intersects or lies inside the rectangle.
    """

    def _inside(p: tuple[float, float]) -> bool:
        return xmin <= p[0] <= xmax and ymin <= p[1] <= ymax

    if _inside(p1) or _inside(p2):
        return True

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    p = [-dx, dx, -dy, dy]
    q = [p1[0] - xmin, xmax - p1[0], p1[1] - ymin, ymax - p1[1]]
    t0, t1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return False
        elif pi < 0:
            t0 = max(t0, qi / pi)
        else:
            t1 = min(t1, qi / pi)
        if t0 > t1:
            return False
    return True


def _arm_collides(
    robot: Any,
    q1: float,
    q2: float,
    obstacle: list[float],
) -> bool:
    """Check whether any arm link intersects the rectangular obstacle.

    Args:
        robot: An :class:`~arco.kinematics.RRRobot` instance.
        q1: First joint angle in radians.
        q2: Second joint angle in radians.
        obstacle: ``[x_min, y_min, x_max, y_max]`` in metres.

    Returns:
        ``True`` if any link segment intersects the rectangle.
    """
    origin, j2, ee = robot.link_segments(q1, q2)
    xmin, ymin, xmax, ymax = obstacle
    return _segment_intersects_rect(
        origin, j2, xmin, ymin, xmax, ymax
    ) or _segment_intersects_rect(j2, ee, xmin, ymin, xmax, ymax)


def build_cspace_occupancy(
    robot: Any,
    obstacle: list[float],
    clearance: float,
    grid_n: int = 120,
) -> tuple[Any, list[list[float]]]:
    """Build a joint-space KDTree occupancy map for the rectangular obstacle.

    Samples a ``grid_n × grid_n`` grid of joint configurations and marks
    those whose arm links intersect the Cartesian obstacle rectangle.

    A single dummy point at the origin is injected when no collision
    configurations are found (e.g. the obstacle is entirely outside the
    workspace), so the KDTree constructor receives a non-empty point cloud.
    In that degenerate case the planner will still find a path because the
    clearance distance keeps the dummy point's neighbourhood small.

    Args:
        robot: An :class:`~arco.kinematics.RRRobot` instance.
        obstacle: ``[x_min, y_min, x_max, y_max]`` in metres.
        clearance: KDTree clearance radius in radians.
        grid_n: Number of samples per joint axis (default 120).

    Returns:
        A tuple ``(occupancy, collision_pts)`` where *occupancy* is the
        :class:`~arco.mapping.KDTreeOccupancy` and *collision_pts* is the
        raw list of ``[q1, q2]`` collision samples used to build it.
    """
    from arco.mapping import KDTreeOccupancy

    theta_samples = np.linspace(-math.pi, math.pi, grid_n)
    collision_pts: list[list[float]] = []
    for q1v in theta_samples:
        for q2v in theta_samples:
            if _arm_collides(robot, float(q1v), float(q2v), obstacle):
                collision_pts.append([float(q1v), float(q2v)])
    if not collision_pts:
        # Obstacle entirely outside workspace — insert a dummy point well
        # outside the valid joint bounds (±π) so that it never interferes
        # with real collision checking or planner paths.
        collision_pts = [[math.pi + 1.0, math.pi + 1.0]]
    return KDTreeOccupancy(collision_pts, clearance=clearance), collision_pts


# ---------------------------------------------------------------------------
# Scene class
# ---------------------------------------------------------------------------


class RRScene:
    """2-D revolute-revolute arm planning scene.

    Builds a joint-space KDTree occupancy map from a Cartesian rectangular
    obstacle, then runs both RRT* and SST planners between start and goal
    joint configurations.

    Args:
        cfg: Configuration dict loaded from ``tools/config/rr.yml``.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        self._robot: Any = None
        self._obstacle: list[float] = []
        self._start_q: np.ndarray = np.zeros(2)
        self._goal_q: np.ndarray = np.zeros(2)
        self._rrt_path: list[np.ndarray] | None = None
        self._sst_path: list[np.ndarray] | None = None
        self._rrt_traj: list[np.ndarray] = []
        self._sst_traj: list[np.ndarray] = []
        self._collision_pts: list[list[float]] = []
        self._rrt_metrics: dict[str, Any] = {
            "steps": 0,
            "nodes": 0,
            "planner_time": 0.0,
            "planned_path_length": 0.0,
            "trajectory_arc_length": 0.0,
            "trajectory_duration": 0.0,
            "path_status": "stalled",
            "optimizer_status": "not-run",
        }
        self._sst_metrics: dict[str, Any] = copy.deepcopy(
            self._rrt_metrics
        )

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, *, progress=None) -> None:
        """Build occupancy map, run both planners, and optimize paths.

        Args:
            progress: Optional callable
                ``(step_name, step_index, total_steps)`` invoked at each
                build milestone for loading-screen feedback.
        """
        import logging

        from arco.kinematics import RRRobot
        from arco.planning.continuous import (
            RRTPlanner,
            SSTPlanner,
            TrajectoryOptimizer,
        )

        _log = logging.getLogger(__name__)
        _total = 5

        # --- Setup robot -----------------------------------------------
        robot = RRRobot(
            l1=float(self._cfg["l1"]), l2=float(self._cfg["l2"])
        )
        self._robot = robot
        obstacle: list[float] = [float(v) for v in self._cfg["obstacle"]]
        self._obstacle = obstacle
        bounds = [tuple(b) for b in self._cfg["bounds"]]

        start_xy: list[float] = [float(v) for v in self._cfg["start_xy"]]
        goal_xy: list[float] = [float(v) for v in self._cfg["goal_xy"]]

        ik_start = robot.inverse_kinematics(start_xy[0], start_xy[1])
        self._start_q = (
            np.array(ik_start[0]) if ik_start else np.array([-2.2, 1.8])
        )
        ik_goal = robot.inverse_kinematics(goal_xy[0], goal_xy[1])
        self._goal_q = (
            np.array(ik_goal[0]) if ik_goal else np.array([1.0, -1.6])
        )

        # --- Build C-space occupancy (uses module-level helpers) -------
        if progress is not None:
            progress("Building C-space occupancy", 1, _total)
        occ, collision_pts = build_cspace_occupancy(
            robot,
            obstacle,
            clearance=float(self._cfg["obstacle_clearance"]),
        )
        self._collision_pts = collision_pts

        # --- RRT* ------------------------------------------------------
        if progress is not None:
            progress("Running RRT*", 2, _total)
        rrt = RRTPlanner(
            occ,
            bounds=bounds,
            max_sample_count=int(self._cfg["rrt_max_sample_count"]),
            step_size=float(self._cfg["step_size"]),
            goal_tolerance=float(self._cfg["goal_tolerance"]),
            collision_check_count=int(self._cfg["collision_check_count"]),
            goal_bias=float(self._cfg["goal_bias"]),
            early_stop=True,
        )
        rrt_t0 = time.perf_counter()
        rrt_result = rrt.plan(self._start_q, self._goal_q)
        rrt_elapsed = time.perf_counter() - rrt_t0
        self._rrt_path = (
            [np.asarray(p) for p in rrt_result] if rrt_result else None
        )
        rrt_len = _polyline_length(self._rrt_path)
        self._rrt_metrics = {
            "steps": len(self._rrt_path) - 1 if self._rrt_path else 0,
            "nodes": getattr(rrt, "_node_count", 0),
            "planner_time": rrt_elapsed,
            "planned_path_length": rrt_len,
            "trajectory_arc_length": 0.0,
            "trajectory_duration": 0.0,
            "path_status": "found" if self._rrt_path else "stalled",
            "optimizer_status": "not-run",
        }
        _log.info(
            "RRT*: %d waypoints, length=%.3f rad",
            len(self._rrt_path) if self._rrt_path else 0,
            rrt_len,
        )

        # --- SST -------------------------------------------------------
        if progress is not None:
            progress("Running SST", 3, _total)
        sst = SSTPlanner(
            occ,
            bounds=bounds,
            max_sample_count=int(self._cfg["sst_max_sample_count"]),
            step_size=float(self._cfg["step_size"]),
            goal_tolerance=float(self._cfg["goal_tolerance"]),
            collision_check_count=int(self._cfg["collision_check_count"]),
            goal_bias=float(self._cfg["goal_bias"]),
            witness_radius=float(self._cfg["witness_radius"]),
            early_stop=True,
        )
        sst_t0 = time.perf_counter()
        sst_result = sst.plan(self._start_q, self._goal_q)
        sst_elapsed = time.perf_counter() - sst_t0
        self._sst_path = (
            [np.asarray(p) for p in sst_result] if sst_result else None
        )
        sst_len = _polyline_length(self._sst_path)
        self._sst_metrics = {
            "steps": len(self._sst_path) - 1 if self._sst_path else 0,
            "nodes": getattr(sst, "_node_count", 0),
            "planner_time": sst_elapsed,
            "planned_path_length": sst_len,
            "trajectory_arc_length": 0.0,
            "trajectory_duration": 0.0,
            "path_status": "found" if self._sst_path else "stalled",
            "optimizer_status": "not-run",
        }
        _log.info(
            "SST: %d waypoints, length=%.3f rad",
            len(self._sst_path) if self._sst_path else 0,
            sst_len,
        )

        # --- Trajectory optimisation -----------------------------------
        if progress is not None:
            progress("Optimizing trajectories", 4, _total)

        optimizer = TrajectoryOptimizer(
            occ,
            cruise_speed=float(self._cfg.get("race_speed", 1.0)),
            weight_time=10.0,
            weight_deviation=1.0,
            weight_velocity=1.0,
            weight_collision=5.0,
            sample_count=1,
            max_iter=200,
        )

        for path, is_rrt in (
            (self._rrt_path, True),
            (self._sst_path, False),
        ):
            if path is None or len(path) < 2:
                continue
            try:
                result = optimizer.optimize(path)
                traj = list(result.states)
                traj_len = _polyline_length(traj)
                dur = (
                    float(sum(result.durations))
                    if result.durations
                    else 0.0
                )
                status = (
                    f"{result.optimizer_status_code}:"
                    f" {result.optimizer_status_text}"
                )
                _log.info(
                    "%s trajectory optimized: cost=%.3f",
                    "RRT*" if is_rrt else "SST",
                    result.cost,
                )
            except Exception as exc:
                _log.warning("Trajectory optimization failed: %s", exc)
                traj = list(path)
                traj_len = _polyline_length(traj)
                dur = 0.0
                status = "error"
            if is_rrt:
                self._rrt_traj = traj
                self._rrt_metrics["trajectory_arc_length"] = traj_len
                self._rrt_metrics["trajectory_duration"] = dur
                self._rrt_metrics["optimizer_status"] = status
            else:
                self._sst_traj = traj
                self._sst_metrics["trajectory_arc_length"] = traj_len
                self._sst_metrics["trajectory_duration"] = dur
                self._sst_metrics["optimizer_status"] = status

        if progress is not None:
            progress("Done", 5, _total)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def robot(self) -> Any:
        """The :class:`~arco.kinematics.RRRobot` instance."""
        return self._robot

    @property
    def obstacle(self) -> list[float]:
        """Rectangular obstacle ``[x_min, y_min, x_max, y_max]``."""
        return self._obstacle

    @property
    def start_q(self) -> np.ndarray:
        """Start joint configuration ``[q1, q2]`` in radians."""
        return self._start_q

    @property
    def goal_q(self) -> np.ndarray:
        """Goal joint configuration ``[q1, q2]`` in radians."""
        return self._goal_q

    @property
    def rrt_path(self) -> list[np.ndarray] | None:
        """Raw RRT* joint-space path, or ``None`` if planning failed."""
        return self._rrt_path

    @property
    def sst_path(self) -> list[np.ndarray] | None:
        """Raw SST joint-space path, or ``None`` if planning failed."""
        return self._sst_path

    @property
    def rrt_traj(self) -> list[np.ndarray]:
        """Optimized RRT* joint-space trajectory."""
        return self._rrt_traj

    @property
    def sst_traj(self) -> list[np.ndarray]:
        """Optimized SST joint-space trajectory."""
        return self._sst_traj

    @property
    def collision_pts(self) -> list[list[float]]:
        """List of ``[q1, q2]`` joint configs that collide with the obstacle."""
        return self._collision_pts

    @property
    def rrt_metrics(self) -> dict[str, Any]:
        """Planning metrics for RRT*."""
        return self._rrt_metrics

    @property
    def sst_metrics(self) -> dict[str, Any]:
        """Planning metrics for SST."""
        return self._sst_metrics
