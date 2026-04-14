"""RRP 3-D SCARA-like robot dual-planner scene.

:class:`RRPScene` builds a 3-D C-space occupancy map for a two-link planar
RR arm elevated by a vertical prismatic joint, then runs both RRT* and SST
planners between a start and goal configuration.

The configuration space is 3-D: ``(q1, q2, z)``.  Collision checking
combines the **planar** arm-link intersection test from the RR scene (both
links at the current Z height against every obstacle's XY footprint) with a
**vertical** Z-range test (the obstacle must span the arm's current Z).

Two obstacle types force combined motion:

* **Pillar obstacles** — full-height box columns.  The revolute joints must
  route around them in the XY plane; the Z joint alone cannot escape.
* **Slab obstacles** — wide horizontal barriers with limited Z extent.  The
  Z joint must find a height corridor; the revolute joints alone cannot escape.

Module-level helpers :func:`_arm_collides_3d` and
:func:`build_cspace_occupancy_3d` are exported so that
``tools/examples/rrp.py`` can import them without duplication.
"""

from __future__ import annotations

import copy
import math
import time
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Collision helpers (exported)
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
    """Parametric clipping of a 2-D segment against an AABB.

    Also returns ``True`` if either endpoint lies inside the rectangle.

    Args:
        p1: Start point ``(x, y)``.
        p2: End point ``(x, y)``.
        xmin: Left bound.
        ymin: Bottom bound.
        xmax: Right bound.
        ymax: Top bound.

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


def _arm_collides_3d(
    robot: Any,
    q1: float,
    q2: float,
    z: float,
    obstacle: list[float],
) -> bool:
    """Check whether any arm link intersects a 3-D box obstacle.

    The arm occupies the XY plane at height *z*.  A box intersects the arm
    only when its Z range includes *z* **and** its XY footprint overlaps at
    least one of the two arm-link segments.

    Args:
        robot: An :class:`~arco.kinematics.RRPRobot` instance.
        q1: First revolute joint angle in radians.
        q2: Second revolute joint angle in radians.
        z: Prismatic joint height in metres.
        obstacle: ``[x_min, y_min, z_min, x_max, y_max, z_max]`` in metres.

    Returns:
        ``True`` if any link segment collides with the obstacle.
    """
    xmin, ymin, zmin, xmax, ymax, zmax = obstacle
    if not (zmin <= z <= zmax):
        return False
    origin, j2, ee = robot.link_segments(q1, q2, z)
    p1 = (origin[0], origin[1])
    p2 = (j2[0], j2[1])
    p3 = (ee[0], ee[1])
    return _segment_intersects_rect(
        p1, p2, xmin, ymin, xmax, ymax
    ) or _segment_intersects_rect(p2, p3, xmin, ymin, xmax, ymax)


def build_cspace_occupancy_3d(
    robot: Any,
    obstacles: list[list[float]],
    clearance: float,
    grid_n: int = 60,
) -> tuple[Any, list[list[float]]]:
    """Build a 3-D joint-space KDTree occupancy map.

    Samples a ``grid_n × grid_n × grid_n`` grid over the joint-space
    bounds ``(q1, q2, z)`` and marks configurations whose arm links
    intersect any 3-D obstacle box.

    Both revolute and prismatic bounds are taken from *robot* properties.

    Args:
        robot: An :class:`~arco.kinematics.RRPRobot` instance.
        obstacles: List of obstacles, each
            ``[x_min, y_min, z_min, x_max, y_max, z_max]``.
        clearance: KDTree clearance radius (blended C-space units).
        grid_n: Samples per joint axis (default 60; total is ``grid_n³``).

    Returns:
        A tuple ``(occupancy, collision_pts)`` where *occupancy* is a
        :class:`~arco.mapping.KDTreeOccupancy` and *collision_pts* is
        the raw list of ``[q1, q2, z]`` collision samples.
    """
    from arco.mapping import KDTreeOccupancy

    theta_samples = np.linspace(-math.pi, math.pi, grid_n)
    z_samples = np.linspace(robot.z_min, robot.z_max, grid_n)
    collision_pts: list[list[float]] = []
    for q1v in theta_samples:
        for q2v in theta_samples:
            for zv in z_samples:
                if any(
                    _arm_collides_3d(
                        robot, float(q1v), float(q2v), float(zv), obs
                    )
                    for obs in obstacles
                ):
                    collision_pts.append([float(q1v), float(q2v), float(zv)])
    if not collision_pts:
        collision_pts = [[math.pi + 1.0, math.pi + 1.0, robot.z_max + 1.0]]
    return KDTreeOccupancy(collision_pts, clearance=clearance), collision_pts


def pick_collision_free_config(
    robot: Any,
    xy: list[float],
    z: float,
    obstacles: list[list[float]],
    fallback: list[float],
) -> Any:
    """Return the first collision-free IK solution for *(xy, z)*.

    Args:
        robot: An :class:`~arco.kinematics.RRPRobot` instance.
        xy: Target end-effector ``[x, y]`` in metres.
        z: Target prismatic height in metres.
        obstacles: List of 3-D obstacles.
        fallback: Default ``[q1, q2, z]`` when IK has no solutions.

    Returns:
        A ``numpy.ndarray`` of shape ``(3,)`` with ``[q1, q2, z]``.
    """
    solutions = robot.inverse_kinematics_xy(xy[0], xy[1])
    for q1, q2 in solutions:
        if not any(
            _arm_collides_3d(robot, q1, q2, z, obs) for obs in obstacles
        ):
            return np.array([q1, q2, z])
    return (
        np.array([solutions[0][0], solutions[0][1], z])
        if solutions
        else np.array(fallback)
    )


# ---------------------------------------------------------------------------
# Scene class
# ---------------------------------------------------------------------------


class RRPScene:
    """3-D SCARA (RRP) robot arm dual-planner scene.

    Builds a 3-D joint-space KDTree occupancy map from a list of 3-D box
    obstacles and runs both RRT* and SST planners between start and goal
    configurations in ``(q1, q2, z)`` joint space.

    Args:
        cfg: Configuration dict loaded from ``tools/config/rrp.yml``.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        self._robot_cfg = cfg.get("robot", cfg)
        self._env_cfg = cfg.get("environment", cfg)
        self._planner_cfg = cfg.get("planner", cfg)
        self._sim_cfg = cfg.get("simulator", cfg)
        self._robot: Any = None
        self._obstacles: list[list[float]] = []
        self._start_q: np.ndarray = np.zeros(3)
        self._goal_q: np.ndarray = np.zeros(3)
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
                milestone for loading-screen feedback.
        """
        import logging

        from arco.kinematics import RRPRobot
        from arco.planning.continuous import (
            RRTPlanner,
            SSTPlanner,
            TrajectoryOptimizer,
            TrajectoryPruner,
        )

        _log = logging.getLogger(__name__)
        _total = 5

        robot = RRPRobot(
            l1=float(self._robot_cfg["l1"]),
            l2=float(self._robot_cfg["l2"]),
            z_min=float(self._robot_cfg["z_min"]),
            z_max=float(self._robot_cfg["z_max"]),
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
        if len(start_position) >= 3:
            start_xy = start_position[:2]
            start_z = float(start_position[2])
        else:
            # Backward compatibility with older configs.
            start_xy = start_position[:2]
            start_z = float(self._env_cfg["start_z"])
        if len(goal_position) >= 3:
            goal_xy = goal_position[:2]
            goal_z = float(goal_position[2])
        else:
            # Backward compatibility with older configs.
            goal_xy = goal_position[:2]
            goal_z = float(self._env_cfg["goal_z"])

        self._start_q = pick_collision_free_config(
            robot, start_xy, start_z, obstacles, [0.0, 0.0, start_z]
        )
        self._goal_q = pick_collision_free_config(
            robot, goal_xy, goal_z, obstacles, [0.0, 0.0, goal_z]
        )
        if any(
            _arm_collides_3d(
                robot,
                float(self._start_q[0]),
                float(self._start_q[1]),
                float(self._start_q[2]),
                obs,
            )
            for obs in obstacles
        ):
            raise ValueError(
                "Start configuration is in collision. "
                "Adjust environment.start_position/obstacles."
            )
        if any(
            _arm_collides_3d(
                robot,
                float(self._goal_q[0]),
                float(self._goal_q[1]),
                float(self._goal_q[2]),
                obs,
            )
            for obs in obstacles
        ):
            raise ValueError(
                "Goal configuration is in collision. "
                "Adjust environment.goal_position/obstacles."
            )

        if progress is not None:
            progress("Building 3-D C-space occupancy", 1, _total)
        grid_n = int(self._planner_cfg.get("cspace_grid_n", 60))
        occ, collision_pts = build_cspace_occupancy_3d(
            robot,
            obstacles,
            clearance=float(self._env_cfg["obstacle_clearance"]),
            grid_n=grid_n,
        )
        self._collision_pts = collision_pts

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
            "RRT*: %d waypoints, length=%.3f",
            len(self._rrt_path) if self._rrt_path else 0,
            rrt_len,
        )

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
            "SST: %d waypoints, length=%.3f",
            len(self._sst_path) if self._sst_path else 0,
            sst_len,
        )

        if progress is not None:
            progress("Optimizing trajectories", 4, _total)

        pruner = TrajectoryPruner(
            occ,
            step_size=np.asarray(
                self._planner_cfg["step_size"], dtype=float
            ),
            collision_check_count=int(
                self._planner_cfg["collision_check_count"]
            ),
        )
        optimizer = TrajectoryOptimizer(
            occ,
            cruise_speed=float(self._sim_cfg.get("race_speed", 0.6)),
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
            path = pruner.prune(path)
            try:
                result = optimizer.optimize(path)
                traj = list(result.states)
                traj_len = _polyline_length(traj)
                dur = float(sum(result.durations)) if result.durations else 0.0
                status = (
                    f"{result.optimizer_status_code}:"
                    f" {result.optimizer_status_text}"
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
    def title(self) -> str:
        """Window caption for the simulator."""
        return "RRT* vs SST — RRP 3-D SCARA arm race"

    @property
    def robot(self) -> Any:
        """The :class:`~arco.kinematics.RRPRobot` instance."""
        return self._robot

    @property
    def obstacles(self) -> list[list[float]]:
        """3-D box obstacles ``[[x1,y1,z1,x2,y2,z2], ...]``."""
        return self._obstacles

    @property
    def start_q(self) -> np.ndarray:
        """Start configuration ``[q1, q2, z]``."""
        return self._start_q

    @property
    def goal_q(self) -> np.ndarray:
        """Goal configuration ``[q1, q2, z]``."""
        return self._goal_q

    @property
    def start(self) -> np.ndarray:
        """Start end-effector 3-D Cartesian position ``[x, y, z]``."""
        if self._robot is None:
            return self._start_q.copy()
        x, y, z = self._robot.forward_kinematics(
            float(self._start_q[0]),
            float(self._start_q[1]),
            float(self._start_q[2]),
        )
        return np.array([x, y, z])

    @property
    def goal(self) -> np.ndarray:
        """Goal end-effector 3-D Cartesian position ``[x, y, z]``."""
        if self._robot is None:
            return self._goal_q.copy()
        x, y, z = self._robot.forward_kinematics(
            float(self._goal_q[0]),
            float(self._goal_q[1]),
            float(self._goal_q[2]),
        )
        return np.array([x, y, z])

    @property
    def bounds(self) -> list[tuple[float, float]]:
        """Planning bounds ``[(q1_min, q1_max), (q2_min, q2_max), (z_min, z_max)]``."""
        return [tuple(b) for b in self._env_cfg["bounds"]]  # type: ignore[return-value]

    @property
    def rrt_path(self) -> list[np.ndarray] | None:
        """Raw RRT* path in joint space, or ``None``."""
        return self._rrt_path

    @property
    def sst_path(self) -> list[np.ndarray] | None:
        """Raw SST path in joint space, or ``None``."""
        return self._sst_path

    @property
    def rrt_traj(self) -> list[np.ndarray]:
        """Optimized RRT* trajectory."""
        return self._rrt_traj

    @property
    def sst_traj(self) -> list[np.ndarray]:
        """Optimized SST trajectory."""
        return self._sst_traj

    @property
    def collision_pts(self) -> list[list[float]]:
        """``[q1, q2, z]`` configurations that collide with any obstacle."""
        return self._collision_pts

    @property
    def rrt_metrics(self) -> dict[str, Any]:
        """Planning metrics for RRT*."""
        return self._rrt_metrics

    @property
    def sst_metrics(self) -> dict[str, Any]:
        """Planning metrics for SST."""
        return self._sst_metrics
