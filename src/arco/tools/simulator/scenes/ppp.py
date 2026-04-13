"""PPP 3-D warehouse dual-planner (RRT* vs SST) race scene.

:class:`PPPScene` builds a larger 3-D industrial warehouse obstacle
environment (60 m × 20 m × 6 m) with ground-level box obstacles.

Three barriers now cross the full bay width in sequence:

    Barrier 1 (tall):   x=[15,17] y=[0,20]  z=[0,2.5]
    Barrier 2 (small):  x=[24,26] y=[0,20]  z=[0,2.0]
    Barrier 3 (split):
      - half south:     x=[38,40] y=[0,10]  z=[0,3.2]
      - half north:     x=[38,40] y=[10,20] z=[0,1.4]

Additional scatter blocks create local detours, and a concave U-obstacle
placed at the far width corner (x=54..59, y=13..17) directly blocks the
final approach to the goal.  Planners that try to cut through the
y=13..17 corridor hit the back wall at x=58..59 and must reroute via the
narrow gap y > 18.5 to reach the goal.

Start: (1, 1, 0) — near the (0, 0, 0) corner.
Goal : (59, 19, 0) — near the (60, 20, 0) corner.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Environment constants (shared with main/ppp.py and tools/examples)
# ---------------------------------------------------------------------------

#: Box extents: (x_min, y_min, z_min, x_max, y_max, z_max).
BOXES: list[tuple[float, float, float, float, float, float]] = [
    # Width-crossing barriers.
    (15.0, 0.0, 0.0, 17.0, 20.0, 2.5),
    (24.0, 0.0, 0.0, 26.0, 20.0, 2.0),
    (38.0, 0.0, 0.0, 40.0, 10.0, 3.2),
    (38.0, 10.0, 0.0, 40.0, 20.0, 1.4),
    # Scatter blocks.
    (5.0, 14.0, 0.0, 8.0, 19.0, 1.8),
    (20.0, 2.0, 0.0, 23.0, 6.0, 2.2),
    (31.0, 13.0, 0.0, 35.0, 18.0, 2.8),
    (45.0, 4.0, 0.0, 48.0, 9.0, 1.6),
    # Concave U-obstacle (opening faces west) placed at the far width corner.
    # The south leg sits at y=13..14; planners that naively enter
    # the y=13..17 corridor hit the back wall and must reroute.
    (54.0, 13.0, 0.0, 59.0, 14.0, 2.8),  # south leg
    (57.0, 14.0, 0.0, 58.0, 16.0, 2.8),  # back wall
]

#: Maximum x-depth (meters) that classifies a box as a crossing barrier.
WALL_MAX_DEPTH: float = 2.5

#: Minimum y-width (meters) that classifies a box as a crossing barrier.
#: Full-width barriers span 20 m and split barriers span 10 m.
WALL_MIN_WIDTH: float = 9.5


def is_wall(box: tuple[float, float, float, float, float, float]) -> bool:
    """Return ``True`` if *box* is a width-crossing barrier.

    A barrier has a narrow x-depth (≤ :data:`WALL_MAX_DEPTH` m) and a
    wide y-span (≥ :data:`WALL_MIN_WIDTH` m). Scatter boxes fail at least
    one of these criteria.

    Args:
        box: Box extents ``(x1, y1, z1, x2, y2, z2)``.

    Returns:
        ``True`` for the blocking wall, ``False`` for scatter boxes.
    """
    return (box[3] - box[0]) <= WALL_MAX_DEPTH and (
        box[4] - box[1]
    ) >= WALL_MIN_WIDTH


#: Planning bounds: [(x_min, x_max), (y_min, y_max), (z_min, z_max)].
BOUNDS: list[tuple[float, float]] = [
    (0.0, 60.0),
    (0.0, 20.0),
    (0.0, 6.0),
]

#: Start position (opposite corner from goal, both at z = 0).
START: np.ndarray = np.array([1.0, 1.0, 0.0])

#: Goal position (opposite corner from start, both at z = 0).
GOAL: np.ndarray = np.array([59.0, 19.0, 0.0])


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _sample_box_surface(
    x1: float,
    y1: float,
    z1: float,
    x2: float,
    y2: float,
    z2: float,
    spacing: float = 0.4,
) -> list[list[float]]:
    """Sample points on all six faces of an axis-aligned box.

    Args:
        x1: Minimum x coordinate.
        y1: Minimum y coordinate.
        z1: Minimum z coordinate.
        x2: Maximum x coordinate.
        y2: Maximum y coordinate.
        z2: Maximum z coordinate.
        spacing: Approximate distance between adjacent sample points (meters).
            The default 0.4 m gives roughly 3–4 samples per metre, which is
            well below the 1.5 m obstacle clearance radius and ensures the
            KDTree can detect any intrusion into the box volume.

    Returns:
        List of ``[x, y, z]`` surface sample points.
    """
    pts: list[list[float]] = []
    xs = np.arange(x1, x2 + spacing, spacing)
    ys = np.arange(y1, y2 + spacing, spacing)
    zs = np.arange(z1, z2 + spacing, spacing)
    for y in ys:
        for z in zs:
            pts += [[x1, float(y), float(z)], [x2, float(y), float(z)]]
    for x in xs:
        for z in zs:
            pts += [[float(x), y1, float(z)], [float(x), y2, float(z)]]
    for x in xs:
        for y in ys:
            pts += [[float(x), float(y), z1], [float(x), float(y), z2]]
    return pts


# ---------------------------------------------------------------------------
# Scene class
# ---------------------------------------------------------------------------


class PPPScene:
    """3-D warehouse race scene for the PPP gantry robot.

    Builds a KDTree occupancy map from the box surfaces, then runs both
    RRT* and SST planners between the start and goal corners of the
    workspace.

    Args:
        cfg: Configuration dict loaded from ``tools/config/ppp.yml``.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        self._planner_cfg = cfg.get("planner", cfg)
        self._sim_cfg = cfg.get("simulator", cfg)
        self._rrt_nodes: list[np.ndarray] = []
        self._sst_nodes: list[np.ndarray] = []
        self._rrt_path: list[np.ndarray] | None = None
        self._sst_path: list[np.ndarray] | None = None
        self._rrt_traj: list[np.ndarray] = []
        self._sst_traj: list[np.ndarray] = []
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
        self._sst_metrics: dict[str, Any] = dict(self._rrt_metrics)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, *, progress=None) -> None:
        """Build the occupancy map, run both planners, and optimize paths.

        Args:
            progress: Optional callable ``(step_name, step_index, total_steps)``
                invoked at each build milestone for loading-screen feedback.

        Imports are deferred so that ``pygame.init()`` can be called
        before this method if needed.
        """
        import logging

        from arco.mapping import KDTreeOccupancy
        from arco.planning.continuous import (
            RRTPlanner,
            SSTPlanner,
            TrajectoryOptimizer,
        )

        _log = logging.getLogger(__name__)
        _total = 5

        def _polyline_length(path: list[np.ndarray] | None) -> float:
            if path is None or len(path) < 2:
                return 0.0
            return sum(
                float(np.linalg.norm(path[i + 1] - path[i]))
                for i in range(len(path) - 1)
            )

        if progress is not None:
            progress("Sampling obstacle surfaces", 1, _total)
        all_pts: list[list[float]] = []
        for box in BOXES:
            all_pts.extend(_sample_box_surface(*box))

        if progress is not None:
            progress("Building occupancy map", 2, _total)
        occ = KDTreeOccupancy(
            all_pts,
            clearance=float(self._planner_cfg["obstacle_clearance"]),
        )

        if progress is not None:
            progress("Running RRT*", 3, _total)
        rrt = RRTPlanner(
            occ,
            bounds=BOUNDS,
            max_sample_count=int(self._planner_cfg["rrt_max_sample_count"]),
            step_size=self._planner_cfg["step_size"],
            goal_tolerance=float(self._planner_cfg["goal_tolerance"]),
            collision_check_count=int(
                self._planner_cfg["collision_check_count"]
            ),
            goal_bias=float(self._planner_cfg["goal_bias"]),
        )
        rrt_t0 = time.perf_counter()
        self._rrt_nodes, _, self._rrt_path = rrt.get_tree(
            START.copy(), GOAL.copy()
        )
        rrt_elapsed = time.perf_counter() - rrt_t0
        self._rrt_metrics.update(
            {
                "steps": max(
                    0,
                    (
                        (len(self._rrt_path) - 1)
                        if self._rrt_path is not None
                        else 0
                    ),
                ),
                "nodes": len(self._rrt_nodes),
                "planner_time": rrt_elapsed,
                "planned_path_length": _polyline_length(self._rrt_path),
                "path_status": (
                    "found" if self._rrt_path is not None else "stalled"
                ),
            }
        )

        if progress is not None:
            progress("Running SST", 4, _total)
        sst = SSTPlanner(
            occ,
            bounds=BOUNDS,
            max_sample_count=int(self._planner_cfg["sst_max_sample_count"]),
            step_size=self._planner_cfg["step_size"],
            goal_tolerance=float(self._planner_cfg["goal_tolerance"]),
            witness_radius=float(self._planner_cfg["witness_radius"]),
            collision_check_count=int(
                self._planner_cfg["collision_check_count"]
            ),
            goal_bias=float(self._planner_cfg["goal_bias"]),
        )
        sst_t0 = time.perf_counter()
        self._sst_nodes, _, self._sst_path = sst.get_tree(
            START.copy(), GOAL.copy()
        )
        sst_elapsed = time.perf_counter() - sst_t0
        self._sst_metrics.update(
            {
                "steps": max(
                    0,
                    (
                        (len(self._sst_path) - 1)
                        if self._sst_path is not None
                        else 0
                    ),
                ),
                "nodes": len(self._sst_nodes),
                "planner_time": sst_elapsed,
                "planned_path_length": _polyline_length(self._sst_path),
                "path_status": (
                    "found" if self._sst_path is not None else "stalled"
                ),
            }
        )

        # --- Trajectory optimization (3-D) --------------------------------
        if progress is not None:
            progress("Optimizing trajectories", 5, _total)
        opt = TrajectoryOptimizer(
            occ,
            cruise_speed=float(self._sim_cfg.get("race_speed", 2.0)),
            weight_time=10.0,
            weight_deviation=1.0,
            weight_velocity=1.0,
            weight_collision=5.0,
            sample_count=1,
            max_iter=50,
        )
        if self._rrt_path is not None:
            try:
                res = opt.optimize(self._rrt_path)
                self._rrt_traj = res.states
                self._rrt_metrics.update(
                    {
                        "trajectory_arc_length": _polyline_length(res.states),
                        "trajectory_duration": sum(res.durations),
                        "path_status": (
                            "found" if res.is_feasible else "stalled"
                        ),
                        "optimizer_status": (
                            f"{res.optimizer_status_code}: "
                            f"{res.optimizer_status_text}"
                        ),
                    }
                )
            except Exception:
                _log.exception(
                    "RRT* TrajectoryOptimizer failed; using raw path."
                )
                self._rrt_traj = list(self._rrt_path)
                self._rrt_metrics.update(
                    {
                        "trajectory_arc_length": _polyline_length(
                            self._rrt_traj
                        ),
                        "trajectory_duration": 0.0,
                        "path_status": "stalled",
                        "optimizer_status": "exception",
                    }
                )
        if self._sst_path is not None:
            try:
                res = opt.optimize(self._sst_path)
                self._sst_traj = res.states
                self._sst_metrics.update(
                    {
                        "trajectory_arc_length": _polyline_length(res.states),
                        "trajectory_duration": sum(res.durations),
                        "path_status": (
                            "found" if res.is_feasible else "stalled"
                        ),
                        "optimizer_status": (
                            f"{res.optimizer_status_code}: "
                            f"{res.optimizer_status_text}"
                        ),
                    }
                )
            except Exception:
                _log.exception(
                    "SST TrajectoryOptimizer failed; using raw path."
                )
                self._sst_traj = list(self._sst_path)
                self._sst_metrics.update(
                    {
                        "trajectory_arc_length": _polyline_length(
                            self._sst_traj
                        ),
                        "trajectory_duration": 0.0,
                        "path_status": "stalled",
                        "optimizer_status": "exception",
                    }
                )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def title(self) -> str:
        """Window caption."""
        return "RRT* vs SST — PPP 3-D warehouse race"

    @property
    def boxes(
        self,
    ) -> list[tuple[float, float, float, float, float, float]]:
        """Box obstacle extents."""
        return BOXES

    @property
    def start(self) -> np.ndarray:
        """Start position (copy)."""
        return START.copy()

    @property
    def goal(self) -> np.ndarray:
        """Goal position (copy)."""
        return GOAL.copy()

    @property
    def bounds(self) -> list[tuple[float, float]]:
        """Workspace bounds."""
        return BOUNDS

    @property
    def rrt_path(self) -> list[np.ndarray] | None:
        """RRT* solution path, or ``None`` if planning failed."""
        return self._rrt_path

    @property
    def sst_path(self) -> list[np.ndarray] | None:
        """SST solution path, or ``None`` if planning failed."""
        return self._sst_path

    @property
    def rrt_traj(self) -> list[np.ndarray]:
        """Optimized RRT* trajectory states, or empty list."""
        return self._rrt_traj

    @property
    def sst_traj(self) -> list[np.ndarray]:
        """Optimized SST trajectory states, or empty list."""
        return self._sst_traj

    @property
    def rrt_metrics(self) -> dict[str, Any]:
        """RRT* planning and trajectory metrics for HUD display."""
        return dict(self._rrt_metrics)

    @property
    def sst_metrics(self) -> dict[str, Any]:
        """SST planning and trajectory metrics for HUD display."""
        return dict(self._sst_metrics)
