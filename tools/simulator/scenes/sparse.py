"""Cul-de-sac dual-planner (RRT* vs SST) race scene.

:class:`SparseScene` builds a 2-D obstacle environment featuring a large
concave U-shaped structure (a *cul-de-sac*) that blocks the direct horizontal
path between start and goal.  Both RRT* and SST are run to completion on the
same map so their paths and exploration trees can be compared in a side-by-side
vehicle race.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pygame
from renderer import (
    bake_sdf_surface,
    draw_endpoints,
    draw_exploration_tree,
    draw_obstacles,
    draw_planned_path,
)
from sim.tracking import VehicleConfig

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_C_BG: tuple[int, int, int] = (28, 28, 35)
_C_OBSTACLE: tuple[int, int, int] = (180, 60, 60)

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
_C_SDF_NEAR: tuple[int, int, int] = (80, 35, 35)

# ---------------------------------------------------------------------------
# Planning constants
# ---------------------------------------------------------------------------
# Both planners share the same start and goal.
_START = np.array([3.0, 25.0])
_GOAL = np.array([47.0, 25.0])
_BOUNDS = [(0.0, 50.0), (0.0, 50.0)]

# Vehicle parameters are identical for a fair race.
_VEHICLE_CONFIG = VehicleConfig(
    max_speed=5.0,
    min_speed=0.0,
    cruise_speed=3.5,
    lookahead_distance=4.0,
    goal_radius=3.0,
    max_turn_rate=math.radians(90.0),
    max_acceleration=4.9,
    max_turn_rate_dot=math.radians(3600.0),
    curvature_gain=0.0,
)


class SparseScene:
    """Dual-planner race scene on a sparse cul-de-sac environment.

    Runs RRT* and SST on the same obstacle map.  The map features a U-shaped
    concave wall that blocks the direct horizontal path — both planners must
    discover a route around it (above or below) before the vehicle race
    begins.

    Args:
        cfg: Parsed sparse configuration dict (from ``sparse.yml``).
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        self._occ: Any = None
        self._sdf_surface: pygame.Surface | None = None

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
        """Build the cul-de-sac map and run both RRT* and SST planners.

        Must be called **after** ``pygame.init()`` so that font calls inside
        any sub-components are safe.
        """
        from arco.planning.continuous import RRTPlanner, SSTPlanner

        self._occ = _build_occupancy(self._cfg)

        rrt = RRTPlanner(
            self._occ,
            bounds=_BOUNDS,
            max_sample_count=int(self._cfg["rrt_max_sample_count"]),
            step_size=float(self._cfg["step_size"]),
            goal_tolerance=float(self._cfg["goal_tolerance"]),
            collision_check_count=int(self._cfg["collision_check_count"]),
            goal_bias=float(self._cfg["goal_bias"]),
        )
        self._rrt_nodes, self._rrt_parent, self._rrt_path = rrt.get_tree(
            _START.copy(), _GOAL.copy()
        )

        sst = SSTPlanner(
            self._occ,
            bounds=_BOUNDS,
            max_sample_count=int(self._cfg["sst_max_sample_count"]),
            step_size=float(self._cfg["step_size"]),
            goal_tolerance=float(self._cfg["goal_tolerance"]),
            collision_check_count=int(self._cfg["collision_check_count"]),
            goal_bias=float(self._cfg["goal_bias"]),
            witness_radius=float(self._cfg["witness_radius"]),
        )
        self._sst_nodes, self._sst_parent, self._sst_path = sst.get_tree(
            _START.copy(), _GOAL.copy()
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def title(self) -> str:
        """Window caption."""
        return "RRT* vs SST — cul-de-sac race"

    @property
    def bg_color(self) -> tuple[int, int, int]:
        """Background fill colour."""
        return _C_BG

    @property
    def world_points(self) -> list[tuple[float, float]]:
        """Bounding-box corners for auto-fitting the full view."""
        return [(0.0, 0.0), (50.0, 50.0)]

    @property
    def vehicle_config(self) -> VehicleConfig:
        """Shared vehicle and controller parameters (identical for both racers)."""
        return _VEHICLE_CONFIG

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

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw_background(
        self,
        surface: pygame.Surface,
        transform: object,
        rrt_revealed: int,
        sst_revealed: int,
    ) -> None:
        """Render the obstacle field, both exploration trees, and paths.

        Args:
            surface: Pygame surface to draw onto.
            transform: World-to-screen callable ``(wx, wy) -> (sx, sy)``.
            rrt_revealed: Number of RRT* tree nodes to display.
            sst_revealed: Number of SST tree nodes to display.
        """
        if self._sdf_surface is None:
            self._sdf_surface = bake_sdf_surface(
                self._occ,
                transform,
                (surface.get_width(), surface.get_height()),
                bg_color=_C_BG,
                near_color=_C_SDF_NEAR,
            )
        surface.blit(
            pygame.transform.smoothscale(
                self._sdf_surface,
                (surface.get_width(), surface.get_height()),
            ),
            (0, 0),
        )
        draw_obstacles(surface, self._occ, transform, color=_C_OBSTACLE)

        # RRT* exploration tree (blue)
        draw_exploration_tree(
            surface,
            self._rrt_nodes,
            self._rrt_parent,
            rrt_revealed,
            transform,
            edge_color=_C_RRT_EDGE,
            node_color=_C_RRT_NODE,
        )
        if rrt_revealed >= self.rrt_total and self._rrt_path is not None:
            draw_planned_path(
                surface, self._rrt_path, transform, color=_C_RRT_PATH
            )

        # SST exploration tree (teal)
        draw_exploration_tree(
            surface,
            self._sst_nodes,
            self._sst_parent,
            sst_revealed,
            transform,
            edge_color=_C_SST_EDGE,
            node_color=_C_SST_NODE,
        )
        if sst_revealed >= self.sst_total and self._sst_path is not None:
            draw_planned_path(
                surface, self._sst_path, transform, color=_C_SST_PATH
            )

        draw_endpoints(
            surface,
            _START,
            _GOAL,
            transform,
            start_color=_C_START,
            goal_color=_C_GOAL,
            bg_color=_C_BG,
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_occupancy(cfg: dict[str, Any]) -> Any:
    """Build the cul-de-sac obstacle environment.

    Constructs a U-shaped concave wall (opening toward the start, closed on
    the goal side) sitting at the horizontal mid-point of the map.  The
    direct path at y = 25 enters the U and hits the closing wall; both
    planners must reroute above (y > 37) or below (y < 13) the structure.

    A small number of sparse random obstacles are scattered in open areas
    for visual variety.

    Layout (50 × 50 m world, clearance = 2.0 m)::

        y=35 ──────── top wall (x ∈ [12, 33]) ────────┐
                                                        │ right wall
        y=15 ──────── bot wall (x ∈ [12, 33]) ────────┘  (x=33)

        START (3, 25)  →→→ direct path hits right wall  →→→ GOAL (47, 25)

    Args:
        cfg: Sparse configuration dict providing ``bounds`` and
            ``obstacle_clearance``.

    Returns:
        A :class:`~arco.mapping.KDTreeOccupancy` ready for collision queries.
    """
    from arco.mapping import KDTreeOccupancy

    x_max = float(cfg["bounds"][0][1])
    y_max = float(cfg["bounds"][1][1])
    mid_y = y_max / 2.0  # 25.0

    spacing = 0.8
    top_y = mid_y + 10.0  # 35.0
    bot_y = mid_y - 10.0  # 15.0
    wall_x_start = 12.0
    wall_x_close = 33.0  # right closing wall

    xs_h = list(np.arange(wall_x_start, wall_x_close + spacing, spacing))
    ys_v = list(np.arange(bot_y, top_y + spacing, spacing))

    top_wall = [[x, top_y] for x in xs_h]
    bot_wall = [[x, bot_y] for x in xs_h]
    right_wall = [[wall_x_close, y] for y in ys_v]

    # Sparse random obstacles — placed well outside the cul-de-sac and the
    # start / goal approach corridors.
    rng = np.random.default_rng(42)
    sparse_pts: list[list[float]] = []
    margin = 4.0
    while len(sparse_pts) < 5:
        p = rng.uniform([margin, margin], [x_max - margin, y_max - margin])
        px, py = float(p[0]), float(p[1])
        # Avoid the cul-de-sac bounding box (+ 4 m buffer)
        if (wall_x_start - 4.0) < px < (wall_x_close + 4.0) and (
            bot_y - 4.0
        ) < py < (top_y + 4.0):
            continue
        # Avoid start approach corridor
        if px < 9.0 and abs(py - mid_y) < 6.0:
            continue
        # Avoid goal approach corridor
        if px > x_max - 9.0 and abs(py - mid_y) < 6.0:
            continue
        sparse_pts.append([px, py])

    all_pts = top_wall + bot_wall + right_wall + sparse_pts
    return KDTreeOccupancy(all_pts, clearance=float(cfg["obstacle_clearance"]))
