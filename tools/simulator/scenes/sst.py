"""SST planner scene for the ARCO unified simulator.

:class:`SSTScene` is structurally identical to :class:`~scenes.rrt.RRTScene`
but uses the :class:`~arco.planning.continuous.SSTPlanner` and a teal colour
palette to visually distinguish the two sampling-based planners.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pygame
from renderer import (
    draw_endpoints,
    draw_exploration_tree,
    draw_obstacles,
    draw_planned_path,
    draw_planning_hud,
)
from sim.scene import SimScene
from sim.tracking import VehicleConfig

# ---------------------------------------------------------------------------
# Colour palette (SST-specific — teal tones for the exploration tree)
# ---------------------------------------------------------------------------
_C_BG: tuple[int, int, int] = (28, 28, 35)
_C_OBSTACLE: tuple[int, int, int] = (160, 60, 60)
_C_TREE_EDGE: tuple[int, int, int] = (60, 160, 140)
_C_TREE_NODE: tuple[int, int, int] = (50, 140, 120)
_C_PATH: tuple[int, int, int] = (230, 170, 30)
_C_START: tuple[int, int, int] = (60, 200, 90)
_C_GOAL: tuple[int, int, int] = (220, 80, 220)

# Vehicle parameters matched to the 50 × 50 m SST planning environment.
_VEHICLE_CONFIG = VehicleConfig(
    max_speed=5.0,
    min_speed=0.0,
    cruise_speed=3.0,
    lookahead_distance=4.0,
    goal_radius=3.0,
    max_turn_rate=math.radians(90.0),
    max_acceleration=4.9,
    max_turn_rate_dot=math.radians(3600.0),
    curvature_gain=0.0,
)


class SSTScene(SimScene):
    """SST planning scene on a 2-D scattered-obstacle environment.

    The background-reveal phase incrementally grows the exploration tree from
    root to leaf before transitioning to vehicle tracking.

    Args:
        sst_config: Parsed SST configuration dict (from ``sst.yml``).
    """

    def __init__(self, sst_config: dict[str, Any]) -> None:
        self._cfg = sst_config
        self._occ: Any = None
        self._tree_nodes: list[Any] = []
        self._tree_parent: dict[int, int | None] = {}
        self._path: list[Any] | None = None
        self._start: Any = None
        self._goal: Any = None
        self._bounds: list[tuple[float, float]] = []

    # ------------------------------------------------------------------
    # SimScene interface
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Build the obstacle environment and run SST.

        Raises:
            RuntimeError: If planner configuration is missing required keys.
        """
        from arco.planning.continuous import SSTPlanner

        self._occ = _build_occupancy(self._cfg)
        bounds = [tuple(b) for b in self._cfg["bounds"]]
        self._bounds = [(float(b[0]), float(b[1])) for b in bounds]

        self._start = np.array([2.0, 2.0])
        self._goal = np.array(
            [
                float(self._cfg["bounds"][0][1]) - 2.0,
                float(self._cfg["bounds"][1][1]) - 2.0,
            ]
        )

        planner = SSTPlanner(
            self._occ,
            bounds=bounds,
            max_sample_count=int(self._cfg["max_sample_count"]),
            step_size=float(self._cfg["step_size"]),
            goal_tolerance=float(self._cfg["goal_tolerance"]),
            collision_check_count=int(self._cfg["collision_check_count"]),
            goal_bias=float(self._cfg["goal_bias"]),
            witness_radius=float(self._cfg["witness_radius"]),
        )
        self._tree_nodes, self._tree_parent, self._path = planner.get_tree(
            self._start, self._goal
        )

    @property
    def title(self) -> str:
        """Human-readable scene label."""
        return "SST"

    @property
    def bg_color(self) -> tuple[int, int, int]:
        """Background fill colour."""
        return _C_BG

    @property
    def world_points(self) -> list[tuple[float, float]]:
        """Bounding-box corners for auto-fitting the full view."""
        return [
            (self._bounds[0][0], self._bounds[1][0]),
            (self._bounds[0][1], self._bounds[1][1]),
        ]

    @property
    def zoom_world_points(self) -> list[tuple[float, float]]:
        """Path bounding box (with 15% padding) for the zoomed view.

        Falls back to the full planning bounds when no path was found.
        """
        if self._path is None:
            return self.world_points
        path_xs = [float(p[0]) for p in self._path]
        path_ys = [float(p[1]) for p in self._path]
        pad = max(
            (max(path_xs) - min(path_xs)) * 0.15,
            (max(path_ys) - min(path_ys)) * 0.15,
            5.0,
        )
        return [
            (min(path_xs) - pad, min(path_ys) - pad),
            (max(path_xs) + pad, max(path_ys) + pad),
        ]

    @property
    def waypoints(self) -> list[tuple[float, float]]:
        """Planned path converted to ``(x, y)`` tuples, or empty list."""
        if self._path is None:
            return []
        return [(float(p[0]), float(p[1])) for p in self._path]

    @property
    def vehicle_config(self) -> VehicleConfig:
        """Vehicle and controller parameters."""
        return _VEHICLE_CONFIG

    @property
    def background_total(self) -> int:
        """Total number of exploration-tree nodes to reveal."""
        return len(self._tree_nodes)

    def draw_background(
        self,
        surface: pygame.Surface,
        transform: object,
        revealed: int,
    ) -> None:
        """Draw the obstacle field, exploration tree, and (if complete) path.

        Args:
            surface: Pygame surface to draw onto.
            transform: World-to-screen callable.
            revealed: Number of tree nodes to show (0 = none, total = all).
        """
        draw_obstacles(surface, self._occ, transform, color=_C_OBSTACLE)
        draw_exploration_tree(
            surface,
            self._tree_nodes,
            self._tree_parent,
            revealed,
            transform,
            edge_color=_C_TREE_EDGE,
            node_color=_C_TREE_NODE,
        )
        if revealed >= self.background_total and self._path is not None:
            draw_planned_path(surface, self._path, transform, color=_C_PATH)
        draw_endpoints(
            surface,
            self._start,
            self._goal,
            transform,
            start_color=_C_START,
            goal_color=_C_GOAL,
            bg_color=_C_BG,
        )

    def draw_background_hud(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        revealed: int,
    ) -> None:
        """Draw the planning-phase HUD showing exploration progress.

        Args:
            surface: Pygame surface to draw onto.
            font: Pygame font for rendering text.
            revealed: Number of tree nodes currently visible.
        """
        draw_planning_hud(
            surface,
            font,
            self.title,
            revealed,
            self.background_total,
            self._path is not None,
        )


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------


def _build_occupancy(cfg: dict[str, Any]) -> Any:
    """Build the demonstration obstacle environment.

    Creates a partial horizontal wall and scattered random obstacles with a
    deterministic seed (same geometry as the RRT scene).

    Args:
        cfg: SST configuration dict providing ``bounds`` and
            ``obstacle_clearance``.

    Returns:
        A :class:`~arco.mapping.KDTreeOccupancy` ready for collision queries.
    """
    from arco.mapping import KDTreeOccupancy

    rng = np.random.default_rng(7)
    x_max = float(cfg["bounds"][0][1])
    y_max = float(cfg["bounds"][1][1])

    wall_pts = [[x, y_max / 2.0] for x in np.arange(0.0, x_max * 0.6, 1.5)] + [
        [x, y_max / 2.0] for x in np.arange(x_max * 0.7, x_max, 1.5)
    ]

    margin = 5.0
    scatter_count = 40
    scatter_pts: list[list[float]] = []
    while len(scatter_pts) < scatter_count:
        p = rng.uniform([margin, margin], [x_max - margin, y_max - margin])
        scatter_pts.append(p.tolist())

    all_pts = wall_pts + scatter_pts
    return KDTreeOccupancy(all_pts, clearance=float(cfg["obstacle_clearance"]))
