"""RRT* planner scene for the ARCO unified simulator.

:class:`RRTScene` builds a 2-D obstacle field, runs the RRT* algorithm to
completion, and then reveals the exploration tree incrementally before
transitioning to the vehicle-tracking phase.  After the tree is fully
revealed, the optimised trajectory (produced by
:class:`~arco.planning.continuous.TrajectoryOptimizer`) is overlaid as a
bright highlight; the raw RRT* path is shown in a dimmer colour beneath it.
The vehicle tracks the optimised trajectory, not the raw path.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pygame
import renderer_gl
from sim.scene import SimScene
from sim.tracking import VehicleConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette (RRT*-specific — blue tones for the exploration tree)
# ---------------------------------------------------------------------------
_C_BG: tuple[int, int, int] = (28, 28, 35)
_C_OBSTACLE: tuple[int, int, int] = (160, 60, 60)
_C_TREE_EDGE: tuple[int, int, int] = (60, 120, 180)
_C_TREE_NODE: tuple[int, int, int] = (50, 100, 160)
_C_PATH: tuple[int, int, int] = (230, 170, 30)  # raw path — kept dimmer
_C_TRAJ: tuple[int, int, int] = (
    255,
    100,
    50,
)  # optimised trajectory — highlighted
_C_START: tuple[int, int, int] = (60, 200, 90)
_C_GOAL: tuple[int, int, int] = (220, 80, 220)
_C_SDF_NEAR: tuple[int, int, int] = (80, 35, 35)

_C_HUD = (220, 220, 220)
_C_HUD_SHADOW = (40, 40, 50)

# Alpha for the raw reference path when the trajectory is drawn on top.
_PATH_ALPHA = 0.35

# World-space ring radii for start/goal markers.
_RING_OUTER = 1.2  # metres
_RING_INNER = 0.6  # metres

# Vehicle parameters matched to the 50 × 50 m RRT planning environment.
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


def _c(t: tuple[int, int, int]) -> tuple[float, float, float]:
    return (t[0] / 255.0, t[1] / 255.0, t[2] / 255.0)


class RRTScene(SimScene):
    """RRT* planning scene on a 2-D scattered-obstacle environment.

    The background-reveal phase incrementally grows the exploration tree from
    root to leaf before transitioning to vehicle tracking.  Once the tree is
    fully revealed, the raw RRT* path is shown dimmed and the two-stage
    optimised trajectory is overlaid as an orange highlight.  The vehicle
    then tracks the optimised trajectory.

    Args:
        rrt_config: Parsed RRT configuration dict (from ``rrt.yml``).
    """

    def __init__(self, rrt_config: dict[str, Any]) -> None:
        self._cfg = rrt_config
        self._occ: Any = None
        self._tree_nodes: list[Any] = []
        self._tree_parent: dict[int, int | None] = {}
        self._path: list[Any] | None = None
        self._traj_states: list[np.ndarray] = []
        self._start: Any = None
        self._goal: Any = None
        self._bounds: list[tuple[float, float]] = []
        self._sdf_tex_id: int | None = None

    # ------------------------------------------------------------------
    # SimScene interface
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Build the obstacle environment, run RRT*, and optimise the path.

        Raises:
            RuntimeError: If planner configuration is missing required keys.
        """
        from arco.planning.continuous import RRTPlanner, TrajectoryOptimizer

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

        planner = RRTPlanner(
            self._occ,
            bounds=bounds,
            max_sample_count=int(self._cfg["max_sample_count"]),
            step_size=float(self._cfg["step_size"]),
            goal_tolerance=float(self._cfg["goal_tolerance"]),
            collision_check_count=int(self._cfg["collision_check_count"]),
            goal_bias=float(self._cfg["goal_bias"]),
        )
        self._tree_nodes, self._tree_parent, self._path = planner.get_tree(
            self._start, self._goal
        )

        # --- Trajectory optimisation -----------------------------------
        if self._path is not None:
            try:
                opt = TrajectoryOptimizer(
                    self._occ,
                    cruise_speed=_VEHICLE_CONFIG.cruise_speed,
                    weight_time=10.0,
                    weight_deviation=1.0,
                    weight_velocity=1.0,
                    weight_collision=5.0,
                    sample_count=2,
                    max_iter=200,
                )
                result = opt.optimize(self._path)
                self._traj_states = result.states
            except Exception:
                logger.exception("TrajectoryOptimizer failed; using raw path.")
                self._traj_states = list(self._path)

    @property
    def title(self) -> str:
        """Human-readable scene label."""
        return "RRT*"

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
        pts = self._traj_states if self._traj_states else self._path
        if pts is None:
            return self.world_points
        path_xs = [float(p[0]) for p in pts]
        path_ys = [float(p[1]) for p in pts]
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
        """Optimised trajectory waypoints as ``(x, y)`` tuples.

        Falls back to the raw path when the optimiser was not run.
        """
        pts = self._traj_states if self._traj_states else self._path
        if pts is None:
            return []
        return [(float(p[0]), float(p[1])) for p in pts]

    @property
    def vehicle_config(self) -> VehicleConfig:
        """Vehicle and controller parameters."""
        return _VEHICLE_CONFIG

    @property
    def background_total(self) -> int:
        """Total number of exploration-tree nodes to reveal."""
        return len(self._tree_nodes)

    def draw_background(self, revealed: int) -> None:
        """Draw the obstacle field, exploration tree, and (if complete) paths.

        When fully revealed, draws the raw RRT* path dimmed and the optimised
        trajectory on top as an orange highlight.

        Args:
            revealed: Number of tree nodes to show (0 = none, total = all).
        """
        x_min = self._bounds[0][0]
        x_max = self._bounds[0][1]
        y_min = self._bounds[1][0]
        y_max = self._bounds[1][1]

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
            self._occ.points, *_c(_C_OBSTACLE), point_size=5.0
        )
        renderer_gl.draw_tree(
            self._tree_nodes,
            self._tree_parent,
            revealed,
            *_c(_C_TREE_EDGE),
            *_c(_C_TREE_NODE),
        )
        if revealed >= self.background_total and self._path is not None:
            # Raw path — dimmed so the trajectory stands out.
            renderer_gl.draw_path(
                self._path,
                *_c(_C_PATH),
                width=1.5,
                alpha=_PATH_ALPHA,
            )
            # Optimised trajectory — bright highlight on top.
            if self._traj_states:
                renderer_gl.draw_path(
                    self._traj_states,
                    *_c(_C_TRAJ),
                    width=3.0,
                )

        # Start / goal rings
        sx, sy = float(self._start[0]), float(self._start[1])
        renderer_gl.draw_ring(sx, sy, _RING_OUTER, _RING_INNER, *_c(_C_START))
        renderer_gl.draw_disc(sx, sy, _RING_INNER, *_c(_C_BG))
        gx, gy = float(self._goal[0]), float(self._goal[1])
        renderer_gl.draw_ring(gx, gy, _RING_OUTER, _RING_INNER, *_c(_C_GOAL))
        renderer_gl.draw_disc(gx, gy, _RING_INNER, *_c(_C_BG))

    def draw_background_hud(
        self,
        font: pygame.font.Font,
        sw: int,
        sh: int,
        revealed: int,
    ) -> None:
        """Draw the planning-phase HUD showing exploration progress.

        Args:
            font: Pygame font for rendering text.
            sw: Screen width in pixels.
            sh: Screen height in pixels.
            revealed: Number of tree nodes currently visible.
        """
        lines = [
            self.title,
            f"Nodes: {revealed}/{self.background_total}",
            f"Path: {'found' if self._path is not None else 'none'}",
        ]
        if revealed >= self.background_total:
            lines.append(
                "Traj: optimised" if self._traj_states else "Traj: raw path"
            )
        line_h = font.get_linesize() + 2
        panel_h = len(lines) * line_h + 8
        panel_w = max(font.size(ln)[0] for ln in lines) + 20
        surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        surf.fill((10, 10, 20, 180))
        y = 4
        for line in lines:
            shadow = font.render(line, True, _C_HUD_SHADOW)
            surf.blit(shadow, (11, y + 1))
            text = font.render(line, True, _C_HUD)
            surf.blit(text, (10, y))
            y += line_h
        renderer_gl.blit_overlay(surf, 0, 0, sw, sh)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_occupancy(cfg: dict[str, Any]) -> Any:
    """Build the demonstration obstacle environment.

    Creates a partial horizontal wall and scattered random obstacles with a
    deterministic seed.

    Args:
        cfg: RRT configuration dict providing ``bounds`` and
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
