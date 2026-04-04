"""SST planner scene for the ARCO unified simulator.

:class:`SSTScene` is structurally identical to :class:`~scenes.rrt.RRTScene`
but uses the :class:`~arco.planning.continuous.SSTPlanner` and a teal colour
palette to visually distinguish the two sampling-based planners.

After the exploration tree is fully revealed the scene shows the raw SST path
for a short period, then switches to the optimized trajectory produced by
:class:`~arco.planning.continuous.TrajectoryOptimizer`.  The vehicle
subsequently tracks the optimized trajectory.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pygame
import renderer_gl
from renderer_gl import check_trajectory_clearance
from sim.scene import SimScene
from sim.tracking import VehicleConfig

# ---------------------------------------------------------------------------
_C_BG: tuple[int, int, int] = (28, 28, 35)
_C_OBSTACLE: tuple[int, int, int] = (160, 60, 60)
_C_TREE_EDGE: tuple[int, int, int] = (60, 160, 140)
_C_TREE_NODE: tuple[int, int, int] = (50, 140, 120)
_C_PATH: tuple[int, int, int] = (230, 170, 30)
_C_TRAJ: tuple[int, int, int] = (100, 230, 100)
_C_START: tuple[int, int, int] = (60, 200, 90)
_C_GOAL: tuple[int, int, int] = (220, 80, 220)
_C_SDF_NEAR: tuple[int, int, int] = (80, 35, 35)

_C_HUD = (220, 220, 220)
_C_HUD_SHADOW = (40, 40, 50)

# World-space ring radii for start/goal markers.
_RING_OUTER = 1.2  # metres
_RING_INNER = 0.6  # metres

# Number of extra "reveal ticks" added after the tree phase.  The first half
# displays the raw planned path; the second half replaces it with the
# optimized trajectory.  The loop's _HOLD_FRAMES pause then follows before
# vehicle tracking starts.
_TRAJ_REVEAL_FRAMES = 1500

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


def _c(t: tuple[int, int, int]) -> tuple[float, float, float]:
    return (t[0] / 255.0, t[1] / 255.0, t[2] / 255.0)


class SSTScene(SimScene):
    """SST planning scene on a 2-D scattered-obstacle environment.

    The background-reveal phase incrementally grows the exploration tree from
    root to leaf.  Once the tree is fully revealed the raw SST path is shown
    for a short period, after which it is replaced by the optimized trajectory
    produced by :class:`~arco.planning.continuous.TrajectoryOptimizer`.  The
    vehicle then tracks the optimized trajectory.

    Args:
        sst_config: Parsed SST configuration dict (from ``sst.yml``).
    """

    def __init__(self, sst_config: dict[str, Any]) -> None:
        self._cfg = sst_config
        self._occ: Any = None
        self._tree_nodes: list[Any] = []
        self._tree_parent: dict[int, int | None] = {}
        self._path: list[Any] | None = None
        self._traj: list[Any] | None = None
        self._tree_total: int = 0
        self._start: Any = None
        self._goal: Any = None
        self._bounds: list[tuple[float, float]] = []
        self._sdf_tex_id: int | None = None
        self._finish_hud_lines: list[str] = []

    # ------------------------------------------------------------------
    # SimScene interface
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Build the obstacle environment, run SST, and optimize the path.

        Runs the SST planner to obtain an initial path, then passes that
        path through :class:`~arco.planning.continuous.TrajectoryOptimizer`
        to produce a shorter, smoother trajectory.

        Raises:
            RuntimeError: If planner configuration is missing required keys.
        """
        from arco.planning.continuous import SSTPlanner, TrajectoryOptimizer

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
        self._tree_total = len(self._tree_nodes)

        # Run the trajectory optimizer on the SST solution path.
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

    def draw_background(self, revealed: int) -> None:
        """Draw the obstacle field, exploration tree, and path overlays.

        During the tree-reveal phase (*revealed* ≤ ``_tree_total``) the
        exploration tree grows incrementally.  Once the tree is complete the
        raw SST path is displayed for the first half of
        ``_TRAJ_REVEAL_FRAMES``, then replaced by the optimized trajectory
        for the second half.

        Args:
            revealed: Number of background ticks elapsed.
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

        if revealed >= self._tree_total:
            traj_phase = revealed - self._tree_total
            if (
                traj_phase > _TRAJ_REVEAL_FRAMES // 2
                and self._traj is not None
            ):
                # Second half of traj-reveal phase: show optimized trajectory.
                renderer_gl.draw_path(self._traj, *_c(_C_TRAJ), width=2.5)
            elif self._path is not None:
                # First half: show the raw planned path.
                renderer_gl.draw_path(self._path, *_c(_C_PATH), width=2.5)

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
        """Draw the planning-phase HUD showing exploration and optimizer progress.

        Args:
            font: Pygame font for rendering text.
            sw: Screen width in pixels.
            sh: Screen height in pixels.
            revealed: Number of background ticks currently visible.
        """
        if revealed <= self._tree_total:
            node_count = revealed
            label = self.title
        else:
            traj_phase = revealed - self._tree_total
            show_traj = (
                traj_phase > _TRAJ_REVEAL_FRAMES // 2
                and self._traj is not None
            )
            node_count = self._tree_total
            label = f"{self.title} — {'trajectory' if show_traj else 'path'}"
        lines = [
            label,
            f"Nodes: {node_count}/{self._tree_total}",
            f"Path: {'found' if self._path is not None else 'none'}",
        ]
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
