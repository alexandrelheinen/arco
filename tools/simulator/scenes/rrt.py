"""RRT* planner scene for the ARCO unified simulator.

:class:`RRTScene` builds a 2-D obstacle field, runs the RRT* algorithm to
completion, and then reveals the exploration tree incrementally before
transitioning to the vehicle-tracking phase.

After the tree is fully revealed the scene shows the raw RRT* path for a
short hold period, then switches to the optimized trajectory produced by
:class:`~arco.planning.continuous.TrajectoryOptimizer`.  The vehicle
subsequently tracks the optimized trajectory.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pygame
from renderer import (
    bake_sdf_surface,
    check_trajectory_clearance,
    draw_endpoints,
    draw_exploration_tree,
    draw_obstacles,
    draw_planned_path,
    draw_planning_hud,
)
from sim.scene import SimScene
from sim.tracking import VehicleConfig

# ---------------------------------------------------------------------------
# Colour palette (RRT*-specific — blue tones for the exploration tree)
# ---------------------------------------------------------------------------
_C_BG: tuple[int, int, int] = (28, 28, 35)
_C_OBSTACLE: tuple[int, int, int] = (160, 60, 60)
_C_TREE_EDGE: tuple[int, int, int] = (60, 120, 180)
_C_TREE_NODE: tuple[int, int, int] = (50, 100, 160)
_C_PATH: tuple[int, int, int] = (230, 170, 30)
_C_TRAJ: tuple[int, int, int] = (100, 230, 100)
_C_START: tuple[int, int, int] = (60, 200, 90)
_C_GOAL: tuple[int, int, int] = (220, 80, 220)
_C_SDF_NEAR: tuple[int, int, int] = (80, 35, 35)

# Number of extra "reveal ticks" added after the tree phase.  The first half
# displays the raw planned path; the second half replaces it with the
# optimized trajectory.  The loop's _HOLD_FRAMES pause then follows before
# vehicle tracking starts.
_TRAJ_REVEAL_FRAMES = 1500

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


class RRTScene(SimScene):
    """RRT* planning scene on a 2-D scattered-obstacle environment.

    The background-reveal phase incrementally grows the exploration tree from
    root to leaf.  Once the tree is fully revealed the raw RRT* path is shown
    for a short period, after which it is replaced by the optimized trajectory
    produced by :class:`~arco.planning.continuous.TrajectoryOptimizer`.  The
    vehicle then tracks the optimized trajectory.

    Args:
        rrt_config: Parsed RRT configuration dict (from ``rrt.yml``).
    """

    def __init__(self, rrt_config: dict[str, Any]) -> None:
        self._cfg = rrt_config
        self._occ: Any = None
        self._tree_nodes: list[Any] = []
        self._tree_parent: dict[int, int | None] = {}
        self._path: list[Any] | None = None
        self._traj: list[Any] | None = None
        self._tree_total: int = 0
        self._start: Any = None
        self._goal: Any = None
        self._bounds: list[tuple[float, float]] = []
        self._sdf_surface: pygame.Surface | None = None
        self._finish_hud_lines: list[str] = []

    # ------------------------------------------------------------------
    # SimScene interface
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Build the obstacle environment, run RRT*, and optimize the path.

        Runs the RRT* planner to obtain an initial path, then passes that
        path through :class:`~arco.planning.continuous.TrajectoryOptimizer`
        to produce a shorter, smoother trajectory.

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
        self._tree_total = len(self._tree_nodes)

        # Run the trajectory optimizer on the RRT* solution path.
        # No model-specific feasibility constraints apply here; the
        # Dubins vehicle used in this scene has no joint or workspace limits.
        if self._path is not None:
            opt = TrajectoryOptimizer(
                self._occ,
                lambda state: True,  # no kinematic constraints for this scene
                spatial_step=float(self._cfg.get("traj_spatial_step", 3.0)),
                weight_time=float(self._cfg.get("traj_weight_time", 1.0)),
                weight_deviation=float(
                    self._cfg.get("traj_weight_deviation", 0.1)
                ),
                weight_collision=float(
                    self._cfg.get("traj_weight_collision", 10.0)
                ),
                deviation_bound=float(
                    self._cfg.get("traj_deviation_bound", 5.0)
                ),
                stage1_population_count=int(
                    self._cfg.get("traj_stage1_population_count", 5)
                ),
                stage1_max_iter=int(self._cfg.get("traj_stage1_max_iter", 30)),
                stage2_max_iter=int(self._cfg.get("traj_stage2_max_iter", 50)),
            )
            self._traj = opt.optimize(self._path)

        # Post-optimization safety check: verify that every trajectory point
        # (or raw path point if the optimizer returned None) has at least
        # traj_safety_distance clearance from the nearest obstacle.
        # Vehicle footprint: 0.5 m × 0.5 m square → 4 × 0.25 m = 1.0 m.
        path_to_check = self._traj if self._traj is not None else self._path
        if path_to_check is not None:
            safety_dist = float(self._cfg.get("traj_safety_distance", 1.0))
            is_safe, min_d = check_trajectory_clearance(
                path_to_check, self._occ, safety_dist
            )
            status = "OK  " if is_safe else "FAIL"
            self._finish_hud_lines = [
                f"Safety: {status}  min {min_d:.2f} m  (thr {safety_dist:.1f} m)",
            ]

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
        ref = self._traj if self._traj is not None else self._path
        if ref is None:
            return self.world_points
        path_xs = [float(p[0]) for p in ref]
        path_ys = [float(p[1]) for p in ref]
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
        """Optimized trajectory as ``(x, y)`` tuples, or planned path as fallback."""
        path = self._traj if self._traj is not None else self._path
        if path is None:
            return []
        return [(float(p[0]), float(p[1])) for p in path]

    @property
    def vehicle_config(self) -> VehicleConfig:
        """Vehicle and controller parameters."""
        return _VEHICLE_CONFIG

    @property
    def background_total(self) -> int:
        """Total background reveal ticks: tree nodes plus trajectory-reveal frames."""
        return self._tree_total + _TRAJ_REVEAL_FRAMES

    @property
    def finish_hud_lines(self) -> list[str]:
        """Safety-clearance report shown when the vehicle reaches the goal."""
        return self._finish_hud_lines

    def draw_background(
        self,
        surface: pygame.Surface,
        transform: object,
        revealed: int,
    ) -> None:
        """Draw the obstacle field, exploration tree, and path overlays.

        During the tree-reveal phase (*revealed* ≤ ``_tree_total``) the
        exploration tree grows incrementally.  Once the tree is complete the
        raw RRT* path is displayed for the first half of
        ``_TRAJ_REVEAL_FRAMES``, then replaced by the optimized trajectory
        for the second half.

        Args:
            surface: Pygame surface to draw onto.
            transform: World-to-screen callable.
            revealed: Number of background ticks elapsed.
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
        draw_exploration_tree(
            surface,
            self._tree_nodes,
            self._tree_parent,
            revealed,
            transform,
            edge_color=_C_TREE_EDGE,
            node_color=_C_TREE_NODE,
        )

        if revealed >= self._tree_total:
            traj_phase = revealed - self._tree_total
            if (
                traj_phase > _TRAJ_REVEAL_FRAMES // 2
                and self._traj is not None
            ):
                # Second half of traj-reveal phase: show optimized trajectory.
                draw_planned_path(
                    surface, self._traj, transform, color=_C_TRAJ
                )
            elif self._path is not None:
                # First half: show the raw planned path.
                draw_planned_path(
                    surface, self._path, transform, color=_C_PATH
                )

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
        """Draw the planning-phase HUD showing exploration and optimizer progress.

        Args:
            surface: Pygame surface to draw onto.
            font: Pygame font for rendering text.
            revealed: Number of background ticks currently visible.
        """
        if revealed <= self._tree_total:
            draw_planning_hud(
                surface,
                font,
                self.title,
                revealed,
                self._tree_total,
                self._path is not None,
            )
        else:
            traj_phase = revealed - self._tree_total
            show_traj = (
                traj_phase > _TRAJ_REVEAL_FRAMES // 2
                and self._traj is not None
            )
            draw_planning_hud(
                surface,
                font,
                f"{self.title} — {'trajectory' if show_traj else 'path'}",
                self._tree_total,
                self._tree_total,
                True,
            )


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
