"""OCC (object-centric control) 2-D dual-planner scene.

:class:`OCCScene` builds a 3-D C-space occupancy map (x, y, psi) for a
2-D rigid body manipulated by mobile actuators, then runs RRT* and SST
planners between a start and goal body pose.

Module-level helpers :func:`_square_collides`, :func:`_circle_collides`,
and :func:`build_cspace_occupancy_2d` are exported so that
``tools/examples/occ.py`` can import them without duplication.
"""

from __future__ import annotations

import copy
import math
import time
from typing import Any

import numpy as np


def _square_collides(
    cx: float,
    cy: float,
    psi: float,
    side: float,
    obstacle: list[float],
) -> bool:
    """Check if a rotated square collides with an AABB obstacle.

    Uses the Separating Axis Theorem with four axes (world X, world Y,
    body X, body Y).

    Args:
        cx: Square centre x in metres.
        cy: Square centre y in metres.
        psi: Square heading in radians.
        side: Side length in metres.
        obstacle: ``[x_min, y_min, x_max, y_max]`` in metres.

    Returns:
        ``True`` if the square overlaps the obstacle AABB.
    """
    xmin, ymin, xmax, ymax = obstacle
    a = side / 2.0
    cos_p = math.cos(psi)
    sin_p = math.sin(psi)
    corners_body = [(-a, -a), (a, -a), (a, a), (-a, a)]
    corners_world = [
        (
            cx + cos_p * bx - sin_p * by,
            cy + sin_p * bx + cos_p * by,
        )
        for bx, by in corners_body
    ]

    # Obstacle corners
    obs_corners = [
        (xmin, ymin),
        (xmax, ymin),
        (xmax, ymax),
        (xmin, ymax),
    ]

    # SAT axes: world X, world Y, body X axis, body Y axis
    axes = [
        (1.0, 0.0),
        (0.0, 1.0),
        (cos_p, sin_p),
        (-sin_p, cos_p),
    ]

    for ax, ay in axes:
        sq_projs = [ax * wx + ay * wy for wx, wy in corners_world]
        obs_projs = [ax * wx + ay * wy for wx, wy in obs_corners]
        if max(sq_projs) < min(obs_projs) or max(obs_projs) < min(sq_projs):
            return False
    return True


def _circle_collides(
    cx: float,
    cy: float,
    radius: float,
    obstacle: list[float],
) -> bool:
    """Check if a circle collides with an AABB obstacle.

    Args:
        cx: Circle centre x in metres.
        cy: Circle centre y in metres.
        radius: Circle radius in metres.
        obstacle: ``[x_min, y_min, x_max, y_max]`` in metres.

    Returns:
        ``True`` if the circle overlaps the obstacle AABB.
    """
    xmin, ymin, xmax, ymax = obstacle
    nearest_x = max(xmin, min(cx, xmax))
    nearest_y = max(ymin, min(cy, ymax))
    dist_sq = (cx - nearest_x) ** 2 + (cy - nearest_y) ** 2
    return dist_sq <= radius**2


def build_cspace_occupancy_2d(
    body_type: str,
    body_size: float,
    obstacles: list[list[float]],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    clearance: float,
    grid_n: int = 40,
    psi_n: int = 16,
) -> tuple[Any, list[list[float]]]:
    """Build a 3-D (x, y, psi) C-space KDTree occupancy map.

    Samples a ``grid_n × grid_n × psi_n`` grid over the C-space and marks
    configurations where the body collides with any obstacle.

    Args:
        body_type: ``"square"`` or ``"circle"``.
        body_size: Side length (square) or radius (circle) in metres.
        obstacles: List of AABB obstacles ``[x_min, y_min, x_max, y_max]``.
        x_range: ``(x_min, x_max)`` planning bounds.
        y_range: ``(y_min, y_max)`` planning bounds.
        clearance: KDTree clearance radius in C-space units.
        grid_n: Samples per Cartesian axis (default 40).
        psi_n: Samples for the heading axis (default 16).

    Returns:
        A tuple ``(occupancy, collision_pts)`` where *occupancy* is a
        :class:`~arco.mapping.KDTreeOccupancy` and *collision_pts* is
        the raw list of ``[x, y, psi]`` collision samples.
    """
    from arco.mapping import KDTreeOccupancy

    x_samples = np.linspace(x_range[0], x_range[1], grid_n)
    y_samples = np.linspace(y_range[0], y_range[1], grid_n)
    psi_samples = np.linspace(-math.pi, math.pi, psi_n, endpoint=False)

    collision_pts: list[list[float]] = []
    for xv in x_samples:
        for yv in y_samples:
            for pv in psi_samples:
                collides = False
                for obs in obstacles:
                    if body_type == "square":
                        if _square_collides(
                            float(xv), float(yv), float(pv), body_size, obs
                        ):
                            collides = True
                            break
                    else:
                        if _circle_collides(
                            float(xv), float(yv), body_size, obs
                        ):
                            collides = True
                            break
                if collides:
                    collision_pts.append([float(xv), float(yv), float(pv)])

    if not collision_pts:
        # Dummy far-away point so KDTreeOccupancy is not empty.
        collision_pts = [[x_range[0] - 1e3, y_range[0] - 1e3, 0.0]]
    return KDTreeOccupancy(collision_pts, clearance=clearance), collision_pts


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


class OCCScene:
    """Object-centric control dual-planner scene.

    Builds a 3-D (x, y, psi) C-space occupancy map for a 2-D rigid body
    (square or circle) and runs RRT* and SST planners between a start and
    goal body pose.  After planning, provides trajectory data for the
    pygame 2-D simulator and physics stepping via
    :class:`~arco.control.ActuatorArray`.

    Args:
        cfg: Configuration dict loaded from ``tools/config/occ.yml``.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        self._body_cfg: dict[str, Any] = cfg.get("body", {})
        self._act_cfg: dict[str, Any] = cfg.get("actuator", {})
        self._env_cfg: dict[str, Any] = cfg.get("environment", {})
        self._planner_cfg: dict[str, Any] = cfg.get("planner", {})
        self._ctrl_cfg: dict[str, Any] = cfg.get("control", {})
        self._sim_cfg: dict[str, Any] = cfg.get("simulator", {})

        self._body: Any = None
        self._actuators: Any = None
        self._obstacles: list[list[float]] = []
        self._start_pose: np.ndarray = np.zeros(3)
        self._goal_pose: np.ndarray = np.zeros(3)
        self._rrt_path: list[np.ndarray] | None = None
        self._sst_path: list[np.ndarray] | None = None
        self._rrt_raw_path: list[np.ndarray] | None = None
        self._sst_raw_path: list[np.ndarray] | None = None
        self._rrt_traj: list[np.ndarray] = []
        self._sst_traj: list[np.ndarray] = []
        self._collision_pts: list[list[float]] = []
        self._rrt_metrics: dict[str, Any] = {
            "steps": 0,
            "nodes": 0,
            "planner_time": 0.0,
            "planned_path_length": 0.0,
            "path_status": "stalled",
        }
        self._sst_metrics: dict[str, Any] = copy.deepcopy(self._rrt_metrics)

    def build(self, *, progress: Any = None) -> None:
        """Build occupancy map, run both planners, and optimize paths.

        Args:
            progress: Optional callable
                ``(step_name, step_index, total_steps)`` invoked at each
                milestone for loading-screen feedback.
        """
        import logging

        from arco.control import ActuatorArray, CircleBody, SquareBody
        from arco.planning.continuous import (
            RRTPlanner,
            SSTPlanner,
            TrajectoryOptimizer,
            TrajectoryPruner,
        )

        _log = logging.getLogger(__name__)
        _total = 5

        body_type: str = str(self._body_cfg.get("type", "square"))
        mass = float(self._body_cfg.get("mass", 5.0))
        if body_type == "square":
            side = float(self._body_cfg.get("side_length", 0.5))
            body = SquareBody(mass=mass, side_length=side)
            body_size = side
        else:
            radius = float(self._body_cfg.get("radius", 0.3))
            body = CircleBody(mass=mass, radius=radius)
            body_size = radius
        self._body = body

        count = int(self._act_cfg.get("count", 3))
        standoff = float(self._act_cfg.get("standoff", 0.05))
        omega = float(self._act_cfg.get("omega", 10.0))
        zeta = float(self._act_cfg.get("zeta", 0.7))
        spring_stiffness = float(self._act_cfg.get("spring_stiffness", 100.0))
        self._actuators = ActuatorArray(
            actuator_count=count,
            standoff=standoff,
            omega=omega,
            zeta=zeta,
            spring_stiffness=spring_stiffness,
        )

        obstacles: list[list[float]] = [
            [float(v) for v in obs]
            for obs in self._env_cfg.get("obstacles", [])
        ]
        self._obstacles = obstacles

        x_range = tuple(
            float(v) for v in self._env_cfg.get("x_range", [-4, 4])
        )
        y_range = tuple(
            float(v) for v in self._env_cfg.get("y_range", [-3, 3])
        )
        start_pose = np.array(
            [float(v) for v in self._env_cfg.get("start_pose", [0, 0, 0])],
            dtype=float,
        )
        goal_pose = np.array(
            [float(v) for v in self._env_cfg.get("goal_pose", [0, 0, 0])],
            dtype=float,
        )
        self._start_pose = start_pose
        self._goal_pose = goal_pose

        bounds = [
            (x_range[0], x_range[1]),
            (y_range[0], y_range[1]),
            (-math.pi, math.pi),
        ]

        if progress is not None:
            progress("Building 3-D C-space occupancy", 1, _total)
        grid_n = int(self._planner_cfg.get("cspace_grid_n", 40))
        occ, collision_pts = build_cspace_occupancy_2d(
            body_type=body_type,
            body_size=body_size,
            obstacles=obstacles,
            x_range=(x_range[0], x_range[1]),
            y_range=(y_range[0], y_range[1]),
            clearance=float(
                self._planner_cfg.get(
                    "cspace_clearance",
                    self._planner_cfg.get("goal_tolerance", 0.25),
                )
            ),
            grid_n=grid_n,
            psi_n=16,
        )
        self._collision_pts = collision_pts

        # step_size may be a scalar or a per-dimension list/array.
        _raw_step = self._planner_cfg.get("step_size", 0.2)
        _step_size: float | np.ndarray = (
            np.array(_raw_step, dtype=float)
            if isinstance(_raw_step, list)
            else float(_raw_step)
        )

        if progress is not None:
            progress("Running RRT*", 2, _total)
        rrt = RRTPlanner(
            occ,
            bounds=bounds,
            max_sample_count=int(
                self._planner_cfg.get("rrt_max_sample_count", 10000)
            ),
            step_size=_step_size,
            goal_tolerance=float(
                self._planner_cfg.get("goal_tolerance", 0.25)
            ),
            collision_check_count=int(
                self._planner_cfg.get("collision_check_count", 5)
            ),
            goal_bias=float(self._planner_cfg.get("goal_bias", 0.10)),
            early_stop=True,
        )
        rrt_t0 = time.perf_counter()
        rrt_result = rrt.plan(start_pose, goal_pose)
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
            "path_status": "found" if self._rrt_path else "stalled",
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
            max_sample_count=int(
                self._planner_cfg.get("sst_max_sample_count", 15000)
            ),
            step_size=_step_size,
            goal_tolerance=float(
                self._planner_cfg.get("goal_tolerance", 0.25)
            ),
            collision_check_count=int(
                self._planner_cfg.get("collision_check_count", 5)
            ),
            goal_bias=float(self._planner_cfg.get("goal_bias", 0.10)),
            witness_radius=float(
                self._planner_cfg.get("witness_radius", 0.50)
            ),
            early_stop=True,
        )
        sst_t0 = time.perf_counter()
        sst_result = sst.plan(start_pose, goal_pose)
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
            "path_status": "found" if self._sst_path else "stalled",
        }
        _log.info(
            "SST: %d waypoints, length=%.3f",
            len(self._sst_path) if self._sst_path else 0,
            sst_len,
        )

        if progress is not None:
            progress("Optimizing trajectories", 4, _total)

        from arco.planning import PlanningPipeline

        pruner = TrajectoryPruner(
            occ,
            step_size=np.asarray(
                self._planner_cfg.get("step_size", [1.0, 1.0, 0.1]),
                dtype=float,
            ),
            collision_check_count=int(
                self._planner_cfg.get("collision_check_count", 5)
            ),
        )
        optimizer = TrajectoryOptimizer(
            occ,
            cruise_speed=float(self._sim_cfg.get("race_speed", 0.3)),
            weight_time=10.0,
            weight_deviation=1.0,
            weight_velocity=1.0,
            weight_collision=5.0,
            sample_count=1,
            max_iter=200,
        )
        pipeline = PlanningPipeline(pruner=pruner, optimizer=optimizer)

        # Snapshot raw paths before pruning overwrites them.
        self._rrt_raw_path = list(self._rrt_path) if self._rrt_path else None
        self._sst_raw_path = list(self._sst_path) if self._sst_path else None

        for path, is_rrt in (
            (self._rrt_path, True),
            (self._sst_path, False),
        ):
            if path is None or len(path) < 2:
                continue
            pr = pipeline.run_from_path(path)
            pruned = pr.pruned_path if pr.pruned_path is not None else path
            traj = pr.trajectory if pr.trajectory else list(pruned)
            if is_rrt:
                self._rrt_path = pruned
                self._rrt_traj = traj
            else:
                self._sst_path = pruned
                self._sst_traj = traj

        if progress is not None:
            progress("Done", 5, _total)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def title(self) -> str:
        """Window caption for the simulator."""
        return "RRT* vs SST — Object-centric control (piano movers)"

    @property
    def body(self) -> Any:
        """The rigid body being manipulated."""
        return self._body

    @property
    def actuators(self) -> Any:
        """The :class:`~arco.control.ActuatorArray` instance."""
        return self._actuators

    @property
    def obstacles(self) -> list[list[float]]:
        """2-D AABB obstacles ``[[x_min, y_min, x_max, y_max], ...]``."""
        return self._obstacles

    @property
    def start_pose(self) -> np.ndarray:
        """Start body pose ``[x, y, psi]``."""
        return self._start_pose

    @property
    def goal_pose(self) -> np.ndarray:
        """Goal body pose ``[x, y, psi]``."""
        return self._goal_pose

    @property
    def rrt_raw_path(self) -> list[np.ndarray] | None:
        """Raw (pre-pruning) RRT* path, or ``None``."""
        return self._rrt_raw_path

    @property
    def sst_raw_path(self) -> list[np.ndarray] | None:
        """Raw (pre-pruning) SST path, or ``None``."""
        return self._sst_raw_path

    @property
    def rrt_path(self) -> list[np.ndarray] | None:
        """Pruned RRT* path (trajectory anchors), or ``None``."""
        return self._rrt_path

    @property
    def sst_path(self) -> list[np.ndarray] | None:
        """Pruned SST path (trajectory anchors), or ``None``."""
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
        """``[x, y, psi]`` configurations that collide with any obstacle."""
        return self._collision_pts

    @property
    def rrt_metrics(self) -> dict[str, Any]:
        """Planning metrics for RRT*."""
        return self._rrt_metrics

    @property
    def sst_metrics(self) -> dict[str, Any]:
        """Planning metrics for SST."""
        return self._sst_metrics
