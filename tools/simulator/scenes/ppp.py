"""PPP 3-D warehouse dual-planner (RRT* vs SST) race scene.

:class:`PPPScene` builds a 3-D industrial warehouse obstacle environment
(20 m × 10 m × 6 m) with ground-level box obstacles.  A full-width blocking
wall at x = 7–9 forces both RRT* and SST to arc over the top of the wall
(z > 4.5 m clearance) rather than trying to squeeze around it in y.

Box obstacles
-------------
All boxes rest on the ground plane (z_min = 0) to reflect the physical
constraint of the PPP mounting geometry.

    Main wall   x=[7, 9]   y=[0, 10]  z=[0, 3]  — blocks full bay width
    Scatter A   x=[2, 3.5] y=[6.5,9.5] z=[0,1.5]
    Scatter B   x=[14,16]  y=[0, 3]   z=[0, 2]
    Scatter C   x=[13,15]  y=[7, 10]  z=[0, 2.5]
    Scatter D   x=[17,18.5] y=[4, 7]  z=[0, 1]

Concave U-obstacle (opening faces west, toward the main wall)
-------------------------------------------------------------
Three boxes form a U-shaped pocket between the main wall and the goal.
Planners that sample inside the pocket discover a dead-end and must
backtrack, exercising the algorithms' concave-obstacle avoidance:

    U south leg  x=[10.0,12.5] y=[2.0, 2.5]  z=[0, 2.5]
    U north leg  x=[10.0,12.5] y=[5.5, 6.0]  z=[0, 2.5]
    U back wall  x=[12.0,12.5] y=[2.5, 5.5]  z=[0, 2.5]

Start: (1, 1, 0) — near the (0, 0, 0) corner.
Goal : (19, 9, 0) — near the (20, 10, 0) corner.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Environment constants (shared with main/ppp.py and tools/examples)
# ---------------------------------------------------------------------------

#: Box extents: (x_min, y_min, z_min, x_max, y_max, z_max).
BOXES: list[tuple[float, float, float, float, float, float]] = [
    (7.0, 0.0, 0.0, 9.0, 10.0, 3.0),
    (2.0, 6.5, 0.0, 3.5, 9.5, 1.5),
    (14.0, 0.0, 0.0, 16.0, 3.0, 2.0),
    (13.0, 7.0, 0.0, 15.0, 10.0, 2.5),
    (17.0, 4.0, 0.0, 18.5, 7.0, 1.0),
    # Concave U-obstacle (opening faces west, toward the main wall).
    # Three boxes form a pocket that traps naive path samples.
    (10.0, 2.0, 0.0, 12.5, 2.5, 2.5),  # south leg
    (10.0, 5.5, 0.0, 12.5, 6.0, 2.5),  # north leg
    (12.0, 2.5, 0.0, 12.5, 5.5, 2.5),  # back wall
]

#: Maximum x-depth (metres) that classifies a box as the main blocking wall.
#: The wall spans dx = 2 m (x=7 to x=9); scatter boxes are wider.
WALL_MAX_DEPTH: float = 2.5

#: Minimum y-width (metres) that classifies a box as the main blocking wall.
#: The wall spans dy = 10 m; scatter boxes are narrower.
WALL_MIN_WIDTH: float = 8.0


def is_wall(box: tuple[float, float, float, float, float, float]) -> bool:
    """Return ``True`` if *box* is the full-width blocking wall.

    The main wall has a narrow x-depth (≤ :data:`WALL_MAX_DEPTH` m) and a
    wide y-span (≥ :data:`WALL_MIN_WIDTH` m).  All scatter boxes fail at
    least one of these criteria.

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
    (0.0, 20.0),
    (0.0, 10.0),
    (0.0, 6.0),
]

#: Start position (opposite corner from goal, both at z = 0).
START: np.ndarray = np.array([1.0, 1.0, 0.0])

#: Goal position (opposite corner from start, both at z = 0).
GOAL: np.ndarray = np.array([19.0, 9.0, 0.0])


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
        spacing: Approximate distance between adjacent sample points (metres).
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
        self._rrt_path: list[np.ndarray] | None = None
        self._sst_path: list[np.ndarray] | None = None
        self._rrt_traj: list[np.ndarray] = []
        self._sst_traj: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, *, progress=None) -> None:
        """Build the occupancy map, run both planners, and optimise paths.

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

        if progress is not None:
            progress("Sampling obstacle surfaces", 1, _total)
        all_pts: list[list[float]] = []
        for box in BOXES:
            all_pts.extend(_sample_box_surface(*box))

        if progress is not None:
            progress("Building occupancy map", 2, _total)
        occ = KDTreeOccupancy(
            all_pts, clearance=float(self._cfg["obstacle_clearance"])
        )

        if progress is not None:
            progress("Running RRT*", 3, _total)
        rrt = RRTPlanner(
            occ,
            bounds=BOUNDS,
            max_sample_count=int(self._cfg["rrt_max_sample_count"]),
            step_size=float(self._cfg["step_size"]),
            goal_tolerance=float(self._cfg["goal_tolerance"]),
            collision_check_count=int(self._cfg["collision_check_count"]),
            goal_bias=float(self._cfg["goal_bias"]),
        )
        _, _, self._rrt_path = rrt.get_tree(START.copy(), GOAL.copy())

        if progress is not None:
            progress("Running SST", 4, _total)
        sst = SSTPlanner(
            occ,
            bounds=BOUNDS,
            max_sample_count=int(self._cfg["sst_max_sample_count"]),
            step_size=float(self._cfg["step_size"]),
            goal_tolerance=float(self._cfg["goal_tolerance"]),
            witness_radius=float(self._cfg["witness_radius"]),
            collision_check_count=int(self._cfg["collision_check_count"]),
            goal_bias=float(self._cfg["goal_bias"]),
        )
        _, _, self._sst_path = sst.get_tree(START.copy(), GOAL.copy())

        # --- Trajectory optimisation (3-D) --------------------------------
        if progress is not None:
            progress("Optimising trajectories", 5, _total)
        opt = TrajectoryOptimizer(
            occ,
            cruise_speed=float(self._cfg.get("race_speed", 2.0)),
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
            except Exception:
                _log.exception(
                    "RRT* TrajectoryOptimizer failed; using raw path."
                )
                self._rrt_traj = list(self._rrt_path)
        if self._sst_path is not None:
            try:
                res = opt.optimize(self._sst_path)
                self._sst_traj = res.states
            except Exception:
                _log.exception(
                    "SST TrajectoryOptimizer failed; using raw path."
                )
                self._sst_traj = list(self._sst_path)

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
        """Optimised RRT* trajectory states, or empty list."""
        return self._rrt_traj

    @property
    def sst_traj(self) -> list[np.ndarray]:
        """Optimised SST trajectory states, or empty list."""
        return self._sst_traj
