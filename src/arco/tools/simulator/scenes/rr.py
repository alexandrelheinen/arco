"""RR 2-D robot arm dual-planner scene.

:class:`RRScene` builds a C-space occupancy map for a two-link planar arm,
then runs both RRT* and SST planners between a start and goal joint
configuration.

The arm navigates around rectangular Cartesian obstacles (one per side).
Planning is performed in joint space (theta1, theta2); resulting paths are
converted to Cartesian end-effector traces via forward kinematics for
visualization.

Module-level helpers :func:`_segment_intersects_rect`, :func:`_arm_collides`,
:func:`build_cspace_occupancy`, and :func:`pick_collision_free_ik` are
exported so that ``tools/examples/rr.py`` can import them without
duplication.
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
    obstacles: list[list[float]],
    bounds: list[tuple[float, float]],
    clearance: float,
    grid_n: int = 120,
) -> tuple[Any, list[list[float]]]:
    """Build a joint-space KDTree occupancy map for a list of obstacles.

    Samples a ``grid_n × grid_n`` grid of joint configurations and marks
    those whose arm links intersect any of the Cartesian obstacle rectangles.
    Both link segments (shoulder→elbow and elbow→end-effector) are checked
    against every obstacle, so neither link may touch a box.

    A single dummy point outside the valid joint bounds is injected when no
    collision configurations are found (e.g. all obstacles are outside the
    workspace), so the KDTree constructor receives a non-empty point cloud.
    In that degenerate case the planner still finds a path because the
    clearance keeps the dummy point's influence negligible.

    Args:
        robot: An :class:`~arco.kinematics.RRRobot` instance.
        obstacles: List of obstacles, each ``[x_min, y_min, x_max, y_max]``.
        bounds: List of joint bounds, each ``(min, max)``.
        clearance: KDTree clearance radius in radians.
        grid_n: Number of samples per joint axis (default 120).

    Returns:
        A tuple ``(occupancy, collision_pts)`` where *occupancy* is the
        :class:`~arco.mapping.KDTreeOccupancy` and *collision_pts* is the
        raw list of ``[q1, q2]`` collision samples used to build it.
    """
    from arco.mapping import KDTreeOccupancy

    q1_samples = np.linspace(bounds[0][0], bounds[0][1], grid_n)
    q2_samples = np.linspace(bounds[1][0], bounds[1][1], grid_n)

    collision_pts: list[list[float]] = []
    for q1v in q1_samples:
        for q2v in q2_samples:
            if any(
                _arm_collides(robot, float(q1v), float(q2v), obs)
                for obs in obstacles
            ):
                collision_pts.append([float(q1v), float(q2v)])
    if not collision_pts:
        # All obstacles outside workspace — insert a dummy point well outside
        # the valid joint bounds (±π) so it never interferes with planning.
        collision_pts = [[math.pi + 1.0, math.pi + 1.0]]
    return KDTreeOccupancy(collision_pts, clearance=clearance), collision_pts


def pick_collision_free_ik(
    robot: Any,
    xy: list[float],
    obstacles: list[list[float]],
    fallback: list[float],
) -> Any:
    """Return the first collision-free IK solution for target position *xy*.

    Iterates through all IK solutions and returns the joint configuration
    of the first one whose arm links do not intersect any obstacle.  If no
    collision-free solution exists (degenerate case) the first available
    solution is returned unchanged.

    Args:
        robot: An :class:`~arco.kinematics.RRRobot` instance.
        xy: Target end-effector position ``[x, y]`` in metres.
        obstacles: List of obstacles, each ``[x_min, y_min, x_max, y_max]``.
        fallback: Default ``[q1, q2]`` used when IK returns no solutions at
            all (i.e. the target is outside the reachable workspace).

    Returns:
        A ``numpy.ndarray`` of shape ``(2,)`` with the joint angles
        ``[q1, q2]`` in radians.
    """
    import numpy as np

    solutions = robot.inverse_kinematics(xy[0], xy[1])
    for sol in solutions:
        q1, q2 = sol
        if not any(_arm_collides(robot, q1, q2, obs) for obs in obstacles):
            return np.array([q1, q2])
    return np.array(solutions[0]) if solutions else np.array(fallback)


# ---------------------------------------------------------------------------
# Scene class
# ---------------------------------------------------------------------------


class RRScene:
    """2-D revolute-revolute arm planning scene.

    Builds a joint-space KDTree occupancy map from a list of Cartesian
    rectangular obstacles, then runs both RRT* and SST planners between
    start and goal joint configurations.  Both arm links are checked against
    every obstacle.

    Args:
        cfg: Configuration dict loaded from ``tools/config/rr.yml``.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        self._robot_cfg = cfg.get("robot", cfg)
        self._env_cfg = cfg.get("environment", cfg)
        self._planner_cfg = cfg.get("planner", cfg)
        self._sim_cfg = cfg.get("simulator", cfg)
        self._robot: Any = None
        self._occ: Any = None
        self._obstacles: list[list[float]] = []
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
        self._sst_metrics: dict[str, Any] = copy.deepcopy(self._rrt_metrics)

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
        )

        _log = logging.getLogger(__name__)
        _total = 5

        # --- Setup robot -----------------------------------------------
        robot = RRRobot(
            l1=float(self._robot_cfg["l1"]),
            l2=float(self._robot_cfg["l2"]),
        )
        self._robot = robot
        obstacles: list[list[float]] = [
            [float(v) for v in obs] for obs in self._env_cfg["obstacles"]
        ]
        self._obstacles = obstacles
        bounds = [tuple(b) for b in self._env_cfg["bounds"]]

        start_position: list[float] = [
            float(v) for v in self._env_cfg["start_position"]
        ]
        goal_position: list[float] = [
            float(v) for v in self._env_cfg["goal_position"]
        ]

        self._start_q = pick_collision_free_ik(
            robot, start_position, obstacles, [-2.2, 1.8]
        )
        self._goal_q = pick_collision_free_ik(
            robot, goal_position, obstacles, [1.0, -1.6]
        )

        # --- Build C-space occupancy (uses module-level helpers) -------
        if progress is not None:
            progress("Building C-space occupancy", 1, _total)
        occ, collision_pts = build_cspace_occupancy(
            robot,
            obstacles,
            bounds,
            clearance=float(self._env_cfg["obstacle_clearance"]),
        )
        self._collision_pts = collision_pts
        self._occ = occ

        # --- RRT* ------------------------------------------------------
        if progress is not None:
            progress("Running RRT*", 2, _total)
        rrt = RRTPlanner(
            occ,
            bounds=bounds,
            max_sample_count=int(self._planner_cfg["rrt_max_sample_count"]),
            step_size=self._planner_cfg["step_size"],
            goal_tolerance=float(self._planner_cfg["goal_tolerance"]),
            collision_check_count=int(
                self._planner_cfg["collision_check_count"]
            ),
            goal_bias=float(self._planner_cfg["goal_bias"]),
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
            max_sample_count=int(self._planner_cfg["sst_max_sample_count"]),
            step_size=self._planner_cfg["step_size"],
            goal_tolerance=float(self._planner_cfg["goal_tolerance"]),
            collision_check_count=int(
                self._planner_cfg["collision_check_count"]
            ),
            goal_bias=float(self._planner_cfg["goal_bias"]),
            witness_radius=float(self._planner_cfg["witness_radius"]),
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

        # --- Pruning + Trajectory optimisation (via PlanningPipeline) -----
        if progress is not None:
            progress("Optimizing trajectories", 4, _total)

        from arco.planning import PlanningPipeline
        from arco.planning.continuous import (
            TrajectoryOptimizer,
            TrajectoryPruner,
        )

        step_size = np.asarray(self._planner_cfg["step_size"], dtype=float)
        pruner = TrajectoryPruner(
            occ,
            step_size=step_size,
            collision_check_count=int(
                self._planner_cfg["collision_check_count"]
            ),
        )
        optimizer = TrajectoryOptimizer(
            occ,
            cruise_speed=float(self._sim_cfg.get("race_speed", 1.0)),
            weight_time=10.0,
            weight_deviation=1.0,
            weight_velocity=1.0,
            weight_collision=5.0,
            sample_count=1,
            max_iter=200,
        )
        pipeline = PlanningPipeline(pruner=pruner, optimizer=optimizer)

        for path, is_rrt in (
            (self._rrt_path, True),
            (self._sst_path, False),
        ):
            if path is None or len(path) < 2:
                continue
            pr = pipeline.run_from_path(path)
            pruned = pr.pruned_path if pr.pruned_path is not None else path
            traj = pr.trajectory if pr.trajectory else list(pruned)
            traj_len = _polyline_length(traj)
            dur = pr.total_duration
            status = pr.optimizer_status
            _log.info(
                "%s trajectory: pruned %d→%d, optimized cost ok, dur=%.2fs",
                "RRT*" if is_rrt else "SST",
                len(path),
                len(pruned),
                dur,
            )
            if is_rrt:
                self._rrt_path = pruned
                self._rrt_traj = traj
                self._rrt_metrics["trajectory_arc_length"] = traj_len
                self._rrt_metrics["trajectory_duration"] = dur
                self._rrt_metrics["optimizer_status"] = status
            else:
                self._sst_path = pruned
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
    def obstacles(self) -> list[list[float]]:
        """List of rectangular obstacles ``[[x_min, y_min, x_max, y_max]]``."""
        return self._obstacles

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

    @property
    def occ(self) -> Any:
        """C-space occupancy map (available after :meth:`build`)."""
        return self._occ
