"""City-neighbourhood dual-planner (RRT* vs SST) race scene.

:class:`SparseScene` builds a 2-D obstacle environment featuring a large
(1280 × 720 m, 16:9) grid-city with 8 columns × 7 rows of city blocks.
The vehicles race along the bottom-left ↔ top-right diagonal: start is the
first street intersection near the bottom-left corner, goal is the first
street intersection near the top-right corner.  No straight diagonal path
exists — both planners must navigate through the street grid.

All layout parameters (block extents, street centrelines, start/goal, world
bounds) are loaded from ``tools/config/obstacles.yml``; planner tuning
parameters (step size, sample counts, …) come from ``tools/config/sparse.yml``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import renderer_gl
from sim.tracking import VehicleConfig

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_C_BG: tuple[int, int, int] = (22, 24, 30)           # dark asphalt
_C_BUILDING: tuple[int, int, int] = (145, 125, 100)   # concrete facade
_C_ROAD_DOT: tuple[int, int, int] = (65, 60, 50)      # faded lane marking

# RRT* — blue family
_C_RRT_EDGE: tuple[int, int, int] = (60, 120, 200)
_C_RRT_NODE: tuple[int, int, int] = (45, 95, 170)
_C_RRT_PATH: tuple[int, int, int] = (130, 190, 255)

# SST — teal family
_C_SST_EDGE: tuple[int, int, int] = (40, 180, 155)
_C_SST_NODE: tuple[int, int, int] = (35, 155, 130)
_C_SST_PATH: tuple[int, int, int] = (100, 240, 210)

_C_START: tuple[int, int, int] = (60, 220, 90)
_C_GOAL: tuple[int, int, int] = (220, 80, 220)
_C_SDF_NEAR: tuple[int, int, int] = (62, 50, 38)      # warm shadow near buildings

# World-space ring radii for start/goal markers.
_RING_OUTER = 1.2  # metres
_RING_INNER = 0.6  # metres

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_road_dots(
    obs_cfg: dict[str, Any],
) -> list[tuple[float, float]]:
    """Build road-marking dot positions from the obstacles config.

    Places dots along every horizontal and vertical street centreline defined
    in *obs_cfg* (``street_y`` and ``street_x``) at the spacing given by
    ``road_dot_spacing``.

    Args:
        obs_cfg: Parsed ``obstacles.yml`` dict.

    Returns:
        List of ``(x, y)`` world positions.
    """
    w = float(obs_cfg["world_width"])
    h = float(obs_cfg["world_height"])
    step = float(obs_cfg["road_dot_spacing"])
    dots: list[tuple[float, float]] = []
    for cy in obs_cfg["street_y"]:
        for x in np.arange(0.0, w, step):
            dots.append((float(x), float(cy)))
    for cx in obs_cfg["street_x"]:
        for y in np.arange(0.0, h, step):
            dots.append((float(cx), float(y)))
    return dots


def _make_blocks(
    obs_cfg: dict[str, Any],
) -> list[tuple[float, float, float, float]]:
    """Build the full list of block rectangles from column/row extents.

    Every (column, row) pair in the Cartesian product of ``block_columns``
    and ``block_rows`` becomes one rectangular obstacle block.

    Args:
        obs_cfg: Parsed ``obstacles.yml`` dict.

    Returns:
        List of ``(x_lo, x_hi, y_lo, y_hi)`` rectangles.
    """
    blocks: list[tuple[float, float, float, float]] = []
    for x_lo, x_hi in obs_cfg["block_columns"]:
        for y_lo, y_hi in obs_cfg["block_rows"]:
            blocks.append((
                float(x_lo), float(x_hi),
                float(y_lo), float(y_hi),
            ))
    return blocks


def _make_vehicle_config(obs_cfg: dict[str, Any]) -> VehicleConfig:
    """Build a VehicleConfig scaled to the world defined in *obs_cfg*.

    Parameters are chosen so the vehicle navigates the city at a realistic
    speed relative to the ~30 m street width.

    Args:
        obs_cfg: Parsed ``obstacles.yml`` dict.

    Returns:
        :class:`~sim.tracking.VehicleConfig` ready for both racers.
    """
    return VehicleConfig(
        max_speed=25.0,
        min_speed=0.0,
        cruise_speed=18.0,
        lookahead_distance=35.0,
        goal_radius=20.0,
        max_turn_rate=math.radians(60.0),
        max_acceleration=4.9,
        max_turn_rate_dot=math.radians(3600.0),
        curvature_gain=0.0,
    )


def _c(t: tuple[int, int, int]) -> tuple[float, float, float]:
    return (t[0] / 255.0, t[1] / 255.0, t[2] / 255.0)


class SparseScene:
    """Dual-planner race scene on a city-neighbourhood environment.

    Runs RRT* and SST on the same 1280 × 720 m obstacle map.  Start and goal
    are placed at the first street intersections on the bottom-left and
    top-right diagonals of the city grid.  No straight diagonal path exists;
    planners must discover a street-grid route connecting the two corners.

    Args:
        cfg: Parsed planner configuration dict (from ``sparse.yml``).
        obs_cfg: Parsed obstacle-layout dict (from ``obstacles.yml``).
    """

    def __init__(self, cfg: dict[str, Any], obs_cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        self._obs_cfg = obs_cfg
        self._occ: Any = None
        self._sdf_tex_id: int | None = None

        # Derived layout (built once from obs_cfg)
        w = float(obs_cfg["world_width"])
        h = float(obs_cfg["world_height"])
        self._bounds = [(0.0, w), (0.0, h)]
        sx, sy = obs_cfg["start"]
        self._start = np.array([float(sx), float(sy)])
        gx, gy = obs_cfg["goal"]
        self._goal = np.array([float(gx), float(gy)])
        self._blocks = _make_blocks(obs_cfg)
        self._road_dots = _make_road_dots(obs_cfg)
        self._vehicle_cfg = _make_vehicle_config(obs_cfg)
        # Marker radii scale with world size
        self._ring_outer = w * 0.014   # ≈ 18 m at 1280 m world
        self._ring_inner = w * 0.007   # ≈  9 m

        # RRT* planner output
        self._rrt_nodes: list[Any] = []
        self._rrt_parent: dict[int, int | None] = {}
        self._rrt_path: list[Any] | None = None

        # SST planner output
        self._sst_nodes: list[Any] = []
        self._sst_parent: dict[int, int | None] = {}
        self._sst_path: list[Any] | None = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Build the city-neighbourhood map and run both RRT* and SST planners.

        Must be called **after** ``pygame.init()`` so that font calls inside
        any sub-components are safe.
        """
        from arco.planning.continuous import RRTPlanner, SSTPlanner

        self._occ = _build_occupancy(self._obs_cfg, self._cfg)

        rrt = RRTPlanner(
            self._occ,
            bounds=self._bounds,
            max_sample_count=int(self._cfg["rrt_max_sample_count"]),
            step_size=float(self._cfg["step_size"]),
            goal_tolerance=float(self._cfg["goal_tolerance"]),
            collision_check_count=int(self._cfg["collision_check_count"]),
            goal_bias=float(self._cfg["goal_bias"]),
        )
        self._rrt_nodes, self._rrt_parent, self._rrt_path = rrt.get_tree(
            self._start.copy(), self._goal.copy()
        )

        sst = SSTPlanner(
            self._occ,
            bounds=self._bounds,
            max_sample_count=int(self._cfg["sst_max_sample_count"]),
            step_size=float(self._cfg["step_size"]),
            goal_tolerance=float(self._cfg["goal_tolerance"]),
            collision_check_count=int(self._cfg["collision_check_count"]),
            goal_bias=float(self._cfg["goal_bias"]),
            witness_radius=float(self._cfg["witness_radius"]),
        )
        self._sst_nodes, self._sst_parent, self._sst_path = sst.get_tree(
            self._start.copy(), self._goal.copy()
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def title(self) -> str:
        """Window caption."""
        return "RRT* vs SST — neighbourhood race"

    @property
    def bg_color(self) -> tuple[int, int, int]:
        """Background fill colour."""
        return _C_BG

    @property
    def world_points(self) -> list[tuple[float, float]]:
        """Bounding-box corners for auto-fitting the full view."""
        w = float(self._obs_cfg["world_width"])
        h = float(self._obs_cfg["world_height"])
        return [(0.0, 0.0), (w, h)]

    @property
    def vehicle_config(self) -> VehicleConfig:
        """Shared vehicle and controller parameters (identical for both racers)."""
        return self._vehicle_cfg

    @property
    def rrt_total(self) -> int:
        """Total number of RRT* exploration-tree nodes."""
        return len(self._rrt_nodes)

    @property
    def sst_total(self) -> int:
        """Total number of SST exploration-tree nodes."""
        return len(self._sst_nodes)

    @property
    def rrt_waypoints(self) -> list[tuple[float, float]]:
        """RRT* solution path as ``(x, y)`` tuples, or empty list."""
        if self._rrt_path is None:
            return []
        return [(float(p[0]), float(p[1])) for p in self._rrt_path]

    @property
    def sst_waypoints(self) -> list[tuple[float, float]]:
        """SST solution path as ``(x, y)`` tuples, or empty list."""
        if self._sst_path is None:
            return []
        return [(float(p[0]), float(p[1])) for p in self._sst_path]

    @property
    def road_dots(self) -> list[tuple[float, float]]:
        """Road-marking dot positions along street centrelines."""
        return self._road_dots

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw_background(
        self,
        rrt_revealed: int,
        sst_revealed: int,
        racing: bool = False,
    ) -> None:
        """Render the obstacle field, both exploration trees, and paths.

        Args:
            rrt_revealed: Number of RRT* tree nodes to display.
            sst_revealed: Number of SST tree nodes to display.
            racing: When ``True`` the planned paths are hidden so the vehicle
                trajectories are the only route highlights visible.
        """
        x_min, x_max = self._bounds[0]
        y_min, y_max = self._bounds[1]

        if self._sdf_tex_id is None:
            self._sdf_tex_id = renderer_gl.bake_sdf_texture(
                self._occ,
                x_min,
                x_max,
                y_min,
                y_max,
                _C_BG,
                _C_SDF_NEAR,
            )
        renderer_gl.draw_sdf_background(
            self._sdf_tex_id, x_min, x_max, y_min, y_max
        )

        renderer_gl.draw_obstacle_points(
            self._road_dots, *_c(_C_ROAD_DOT), point_size=3.0
        )

        renderer_gl.draw_obstacle_points(
            self._occ.points, *_c(_C_BUILDING), point_size=5.0
        )

        renderer_gl.draw_tree(
            self._rrt_nodes,
            self._rrt_parent,
            rrt_revealed,
            *_c(_C_RRT_EDGE),
            *_c(_C_RRT_NODE),
        )
        if (
            not racing
            and rrt_revealed >= self.rrt_total
            and self._rrt_path is not None
        ):
            renderer_gl.draw_path(self._rrt_path, *_c(_C_RRT_PATH), width=2.5)

        renderer_gl.draw_tree(
            self._sst_nodes,
            self._sst_parent,
            sst_revealed,
            *_c(_C_SST_EDGE),
            *_c(_C_SST_NODE),
        )
        if (
            not racing
            and sst_revealed >= self.sst_total
            and self._sst_path is not None
        ):
            renderer_gl.draw_path(self._sst_path, *_c(_C_SST_PATH), width=2.5)

        sx, sy = float(self._start[0]), float(self._start[1])
        renderer_gl.draw_ring(
            sx, sy, self._ring_outer, self._ring_inner, *_c(_C_START)
        )
        renderer_gl.draw_disc(sx, sy, self._ring_inner, *_c(_C_BG))
        gx, gy = float(self._goal[0]), float(self._goal[1])
        renderer_gl.draw_ring(
            gx, gy, self._ring_outer, self._ring_inner, *_c(_C_GOAL)
        )
        renderer_gl.draw_disc(gx, gy, self._ring_inner, *_c(_C_BG))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_occupancy(
    obs_cfg: dict[str, Any],
    cfg: dict[str, Any],
) -> Any:
    """Build the city-neighbourhood obstacle environment.

    Samples every block rectangle defined in *obs_cfg* (``block_columns``
    × ``block_rows``) on a regular grid to produce a dense point cloud for
    the KDTree occupancy model.  Street corridors remain obstacle-free.

    Layout (1280 × 720 m, street width ≈ 30 m)::

        8 columns × 7 rows of city blocks fill the space.
        START: first street crossing from bottom-left → (220.1,  148.5)
        GOAL:  first street crossing from top-right   → (1047.3, 582.4)
        No straight diagonal path exists; routes thread through the street
        grid.

    Args:
        obs_cfg: Obstacle-layout dict (from ``obstacles.yml``).
        cfg: Planner configuration dict providing ``obstacle_clearance``.

    Returns:
        A :class:`~arco.mapping.KDTreeOccupancy` ready for collision queries.
    """
    from arco.mapping import KDTreeOccupancy

    spacing = float(obs_cfg["obstacle_sampling_spacing"])
    blocks = _make_blocks(obs_cfg)
    all_pts: list[list[float]] = []
    for x_lo, x_hi, y_lo, y_hi in blocks:
        xs = np.arange(x_lo, x_hi + spacing, spacing)
        ys = np.arange(y_lo, y_hi + spacing, spacing)
        for x in xs:
            for y in ys:
                all_pts.append([float(x), float(y)])
    return KDTreeOccupancy(all_pts, clearance=float(cfg["obstacle_clearance"]))

