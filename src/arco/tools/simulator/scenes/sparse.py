"""City-neighborhood dual-planner (RRT* vs SST) race scene.

:class:`CityScene` builds a 2-D obstacle environment on a 1280 × 720 m
procedural triangular road network.  A jittered hexagonal lattice is
triangulated with scipy Delaunay; every surviving edge (≤ 1.7 × mean-edge-
length) becomes a 30 m wide road corridor.  The interior of each triangular
city block is derived as the set of points more than 15 m from every road
centerline, and is filled with a dense KDTree obstacle point cloud.

Start and goal are auto-placed at the two mesh nodes with the largest
separation, guaranteeing a non-trivial path problem.

After the exploration trees are fully revealed, the raw planned paths are
shown dimmed and the optimized trajectories (produced by
:class:`~arco.planning.continuous.TrajectoryOptimizer`) are overlaid as
bright highlights.  The vehicles track the optimized trajectories.

All layout parameters (world size, mesh geometry, road width, obstacle
sampling) are loaded from ``tools/config/city.yml`` under ``world``;
planner tuning parameters (step size, sample counts, ...) come from the
same file under ``planner``.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import numpy as np
from scipy.spatial import Delaunay as _Delaunay

from arco.config.palette import (
    annotation_rgb,
    layer_rgb,
    obstacle_rgb,
    ui_rgb,
)
from arco.tools.simulator import renderer_gl
from arco.tools.simulator.sim.tracking import VehicleConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color palette — derived from arco.config.palette
# ---------------------------------------------------------------------------
_C_BG: tuple[int, int, int] = ui_rgb("background")
_C_BUILDING: tuple[int, int, int] = obstacle_rgb()
_C_ROAD_DOT: tuple[int, int, int] = ui_rgb("road_dot")

# RRT* — blue family
_C_RRT_EDGE: tuple[int, int, int] = layer_rgb("rrt", "tree")
_C_RRT_NODE: tuple[int, int, int] = layer_rgb("rrt", "tree")
_C_RRT_PATH: tuple[int, int, int] = layer_rgb("rrt", "path")
_C_RRT_TRAJ: tuple[int, int, int] = layer_rgb("rrt", "trajectory")
_C_RRT_PRUNED: tuple[int, int, int] = layer_rgb("rrt", "pruned")
_C_RRT_VEHICLE: tuple[int, int, int] = layer_rgb("rrt", "vehicle")

# SST — green family
_C_SST_EDGE: tuple[int, int, int] = layer_rgb("sst", "tree")
_C_SST_NODE: tuple[int, int, int] = layer_rgb("sst", "tree")
_C_SST_PATH: tuple[int, int, int] = layer_rgb("sst", "path")
_C_SST_TRAJ: tuple[int, int, int] = layer_rgb("sst", "trajectory")
_C_SST_PRUNED: tuple[int, int, int] = layer_rgb("sst", "pruned")
_C_SST_VEHICLE: tuple[int, int, int] = layer_rgb("sst", "vehicle")

_C_START: tuple[int, int, int] = annotation_rgb(dark_bg=True)
_C_GOAL: tuple[int, int, int] = annotation_rgb(dark_bg=True)
_C_SDF_NEAR: tuple[int, int, int] = ui_rgb("road_sdf")
_C_BARRIER: tuple[int, int, int] = ui_rgb("barrier")
_C_HUD: tuple[int, int, int] = ui_rgb("hud_text")
_C_HUD_DIM: tuple[int, int, int] = ui_rgb("hud_dim")
_C_HUD_SHADOW: tuple[int, int, int] = ui_rgb("hud_shadow")
_C_HUD_WINNER: tuple[int, int, int] = ui_rgb("hud_winner")
_C_HUD_TIE: tuple[int, int, int] = ui_rgb("hud_tie")

# Alpha for the raw reference paths when trajectories are drawn on top.
_PATH_ALPHA = 0.3

# World-space ring radii for start/goal markers.
_RING_OUTER = 1.2  # meters
_RING_INNER = 0.6  # meters

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_neighborhood_mesh(
    obs_cfg: dict[str, Any],
) -> tuple[
    list[tuple[float, float]],
    list[tuple[int, int]],
]:
    """Generate a jittered hex-lattice Delaunay mesh for the neighborhood.

    Replicates the algorithm from ``tools/graph/generator.py`` (ring
    generator) but without holes and without S-curve waypoints — here we
    only need the road-network skeleton (nodes + edges).

    Args:
        obs_cfg: Parsed city ``world`` configuration dict. Must contain
            ``world_width``, ``world_height``, ``mean_edge_length``,
            ``jitter_sigma``, ``max_edge_factor``, and ``seed``.

    Returns:
        A ``(positions, edges)`` pair where *positions* is a list of
        ``(x, y)`` node coordinates and *edges* is a list of
        ``(u, v)`` index pairs (u < v, both pruned to at most
        ``max_edge_factor × mean_edge_length``).
    """
    width = float(obs_cfg["world_width"])
    height = float(obs_cfg["world_height"])
    mean_edge = float(obs_cfg["mean_edge_length"])
    sigma = float(obs_cfg.get("jitter_sigma", 0.15))
    max_factor = float(obs_cfg.get("max_edge_factor", 1.7))
    seed = obs_cfg.get("seed", None)

    rng = np.random.default_rng(seed)

    # ── Jittered hexagonal lattice ─────────────────────────────────────────
    row_spacing = mean_edge * math.sqrt(3.0) / 2.0
    col_spacing = mean_edge
    positions: list[tuple[float, float]] = []
    row = 0
    y = 0.0
    while y <= height:
        col_offset = col_spacing / 2.0 if (row % 2 == 1) else 0.0
        x = col_offset
        while x <= width:
            jx = float(rng.normal(0.0, sigma * mean_edge))
            jy = float(rng.normal(0.0, sigma * mean_edge))
            px = float(np.clip(x + jx, 0.0, width))
            py = float(np.clip(y + jy, 0.0, height))
            positions.append((px, py))
            x += col_spacing
        y += row_spacing
        row += 1

    # ── Delaunay triangulation ─────────────────────────────────────────────
    pts = np.array(positions)
    tri = _Delaunay(pts)

    edge_set: set[tuple[int, int]] = set()
    for simplex in tri.simplices:
        a, b, c = int(simplex[0]), int(simplex[1]), int(simplex[2])
        for u, v in ((a, b), (b, c), (a, c)):
            edge_set.add((min(u, v), max(u, v)))

    # ── Prune long edges ───────────────────────────────────────────────────
    max_edge_len = max_factor * mean_edge
    edges: list[tuple[int, int]] = [
        (u, v)
        for u, v in edge_set
        if math.hypot(
            positions[u][0] - positions[v][0],
            positions[u][1] - positions[v][1],
        )
        <= max_edge_len
    ]

    return positions, edges


def _find_farthest_pair(
    positions: list[tuple[float, float]],
) -> tuple[int, int]:
    """Return indices of the two nodes with the greatest Euclidean distance.

    Args:
        positions: List of ``(x, y)`` node coordinates.

    Returns:
        ``(i, j)`` index pair (i < j) of the farthest nodes.
    """
    best_dist = -1.0
    best_i, best_j = 0, 1
    n = len(positions)
    for i in range(n):
        xi, yi = positions[i]
        for j in range(i + 1, n):
            d = math.hypot(xi - positions[j][0], yi - positions[j][1])
            if d > best_dist:
                best_dist = d
                best_i, best_j = i, j
    return best_i, best_j


def _make_road_dots(
    positions: list[tuple[float, float]],
    edges: list[tuple[int, int]],
    obs_cfg: dict[str, Any],
) -> list[tuple[float, float]]:
    """Build road-marking dot positions along every mesh road edge.

    Samples equally spaced dots along each surviving Delaunay edge at the
    spacing given by ``road_dot_spacing``.

    Args:
        positions: Node coordinates from :func:`_generate_neighborhood_mesh`.
        edges: Edge index pairs from :func:`_generate_neighborhood_mesh`.
        obs_cfg: Parsed city ``world`` configuration dict.

    Returns:
        List of ``(x, y)`` world positions.
    """
    step = float(obs_cfg.get("road_dot_spacing", 5.0))
    dots: list[tuple[float, float]] = []
    for u, v in edges:
        ax, ay = positions[u]
        bx, by = positions[v]
        length = math.hypot(bx - ax, by - ay)
        n_steps = max(1, int(length / step))
        for k in range(1, n_steps):
            t = k / n_steps
            dots.append((ax + t * (bx - ax), ay + t * (by - ay)))
    return dots


def _make_vehicle_config(obs_cfg: dict[str, Any]) -> VehicleConfig:
    """Build a VehicleConfig scaled to the world defined in *obs_cfg*.

    Parameters are chosen so a small 50 cm robotic car navigates 10 m-wide
    neighborhood roads at a realistic pace.

    Args:
        obs_cfg: Parsed city ``world`` configuration dict.

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


class CityScene:
    """Dual-planner race scene on a procedural triangular neighborhood.

    Runs RRT* and SST on the same 1280 × 720 m obstacle map generated from a
    jittered hexagonal lattice triangulated with scipy Delaunay.  Every
    surviving edge becomes a 30 m wide road; the interior of each triangular
    city block is filled with KDTree obstacle points.  Start and goal are
    placed at the two mesh nodes with the greatest separation.

    Args:
        cfg: Parsed planner configuration dict (from ``city.yml`` ``planner``).
        obs_cfg: Parsed obstacle-layout dict (from ``city.yml`` ``world``).
    """

    def __init__(self, cfg: dict[str, Any], obs_cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        self._obs_cfg = obs_cfg
        self._occ: Any = None
        self._sdf_tex_id: int | None = None

        # Generate the triangular road-network mesh.
        self._mesh_positions, self._mesh_edges = _generate_neighborhood_mesh(
            obs_cfg
        )

        # Derived layout from mesh
        w = float(obs_cfg["world_width"])
        h = float(obs_cfg["world_height"])
        self._bounds = [(0.0, w), (0.0, h)]

        # Start and goal: farthest pair of mesh nodes.
        si, gi = _find_farthest_pair(self._mesh_positions)
        self._start = np.array(self._mesh_positions[si], dtype=float)
        self._goal = np.array(self._mesh_positions[gi], dtype=float)

        self._road_dots = _make_road_dots(
            self._mesh_positions, self._mesh_edges, obs_cfg
        )
        self._vehicle_cfg = _make_vehicle_config(obs_cfg)
        # Marker radii: fixed at ~14 m outer / 7 m inner (visible at 1280 m world).
        self._ring_outer = 14.0
        self._ring_inner = 7.0

        # RRT* planner output
        self._rrt_nodes: list[Any] = []
        self._rrt_parent: dict[int, int | None] = {}
        self._rrt_path: list[Any] | None = None
        self._rrt_traj_states: list[Any] = []
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

        # SST planner output
        self._sst_nodes: list[Any] = []
        self._sst_parent: dict[int, int | None] = {}
        self._sst_path: list[Any] | None = None
        self._sst_traj_states: list[Any] = []
        self._sst_metrics: dict[str, Any] = dict(self._rrt_metrics)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, *, progress=None) -> None:
        """Build the neighborhood map, run both planners, and optimize.

        Args:
            progress: Optional callable ``(step_name, step_index, total_steps)``
                invoked at each build milestone for loading-screen feedback.
        """
        _total = 5

        def _polyline_length(path: list[Any] | None) -> float:
            if path is None or len(path) < 2:
                return 0.0
            return sum(
                float(
                    np.linalg.norm(np.array(path[i + 1]) - np.array(path[i]))
                )
                for i in range(len(path) - 1)
            )

        from arco.planning.continuous import (
            RRTPlanner,
            SSTPlanner,
            TrajectoryOptimizer,
            TrajectoryPruner,
        )

        if progress is not None:
            progress("Generating triangular mesh", 1, _total)
        # Mesh was already generated in __init__; the expensive step here is
        # building the KDTree occupancy map from it.
        if progress is not None:
            progress("Building occupancy map", 2, _total)
        self._occ = _build_occupancy(
            self._mesh_positions, self._mesh_edges, self._obs_cfg, self._cfg
        )

        if progress is not None:
            progress("Running RRT*", 3, _total)
        rrt = RRTPlanner(
            self._occ,
            bounds=self._bounds,
            max_sample_count=int(self._cfg["rrt_max_sample_count"]),
            step_size=self._cfg["step_size"],
            goal_tolerance=float(self._cfg["goal_tolerance"]),
            collision_check_count=int(self._cfg["collision_check_count"]),
            goal_bias=float(self._cfg["goal_bias"]),
        )
        rrt_t0 = time.perf_counter()
        self._rrt_nodes, self._rrt_parent, self._rrt_path = rrt.get_tree(
            self._start.copy(), self._goal.copy()
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
            self._occ,
            bounds=self._bounds,
            max_sample_count=int(self._cfg["sst_max_sample_count"]),
            step_size=self._cfg["step_size"],
            goal_tolerance=float(self._cfg["goal_tolerance"]),
            collision_check_count=int(self._cfg["collision_check_count"]),
            goal_bias=float(self._cfg["goal_bias"]),
            witness_radius=float(self._cfg["witness_radius"]),
        )
        sst_t0 = time.perf_counter()
        self._sst_nodes, self._sst_parent, self._sst_path = sst.get_tree(
            self._start.copy(), self._goal.copy()
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

        # --- Trajectory optimization (scaled for the large world) --------
        if progress is not None:
            progress("Optimizing trajectories", 5, _total)
        cruise = self._vehicle_cfg.cruise_speed
        pruner = TrajectoryPruner(
            self._occ,
            collision_check_count=int(self._cfg["collision_check_count"]),
        )
        if self._rrt_path is not None:
            self._rrt_path = pruner.prune(self._rrt_path)
        if self._sst_path is not None:
            self._sst_path = pruner.prune(self._sst_path)
        opt = TrajectoryOptimizer(
            self._occ,
            cruise_speed=cruise,
            weight_time=10.0,
            weight_deviation=1.0,
            weight_velocity=1.0,
            weight_collision=5.0,
            sample_count=0,
            max_iter=30,
        )
        if self._rrt_path is not None:
            try:
                result = opt.optimize(self._rrt_path)
                self._rrt_traj_states = result.states
                self._rrt_metrics.update(
                    {
                        "trajectory_arc_length": _polyline_length(
                            result.states
                        ),
                        "trajectory_duration": sum(result.durations),
                        "path_status": (
                            "found" if result.is_feasible else "stalled"
                        ),
                        "optimizer_status": (
                            f"{result.optimizer_status_code}: "
                            f"{result.optimizer_status_text}"
                        ),
                    }
                )
            except Exception:
                logger.exception(
                    "RRT* TrajectoryOptimizer failed; using raw path."
                )
                self._rrt_traj_states = list(self._rrt_path)
                self._rrt_metrics.update(
                    {
                        "trajectory_arc_length": _polyline_length(
                            self._rrt_traj_states
                        ),
                        "trajectory_duration": 0.0,
                        "path_status": "stalled",
                        "optimizer_status": "exception",
                    }
                )
        if self._sst_path is not None:
            try:
                result = opt.optimize(self._sst_path)
                self._sst_traj_states = result.states
                self._sst_metrics.update(
                    {
                        "trajectory_arc_length": _polyline_length(
                            result.states
                        ),
                        "trajectory_duration": sum(result.durations),
                        "path_status": (
                            "found" if result.is_feasible else "stalled"
                        ),
                        "optimizer_status": (
                            f"{result.optimizer_status_code}: "
                            f"{result.optimizer_status_text}"
                        ),
                    }
                )
            except Exception:
                logger.exception(
                    "SST TrajectoryOptimizer failed; using raw path."
                )
                self._sst_traj_states = list(self._sst_path)
                self._sst_metrics.update(
                    {
                        "trajectory_arc_length": _polyline_length(
                            self._sst_traj_states
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
        return "RRT* vs SST — neighborhood race"

    @property
    def bg_color(self) -> tuple[int, int, int]:
        """Background fill color."""
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
        """Optimized RRT* trajectory as ``(x, y)`` tuples, or empty list."""
        pts = (
            self._rrt_traj_states if self._rrt_traj_states else self._rrt_path
        )
        if pts is None:
            return []
        return [(float(p[0]), float(p[1])) for p in pts]

    @property
    def sst_waypoints(self) -> list[tuple[float, float]]:
        """Optimized SST trajectory as ``(x, y)`` tuples, or empty list."""
        pts = (
            self._sst_traj_states if self._sst_traj_states else self._sst_path
        )
        if pts is None:
            return []
        return [(float(p[0]), float(p[1])) for p in pts]

    @property
    def road_dots(self) -> list[tuple[float, float]]:
        """Road-marking dot positions along street centerlines."""
        return self._road_dots

    @property
    def rrt_metrics(self) -> dict[str, Any]:
        """RRT* planning and trajectory metrics for HUD display."""
        return dict(self._rrt_metrics)

    @property
    def sst_metrics(self) -> dict[str, Any]:
        """SST planning and trajectory metrics for HUD display."""
        return dict(self._sst_metrics)

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

        When *racing* is ``False`` the full world is drawn: SDF clearance
        heatmap, road markings, buildings, barriers, exploration trees, raw
        planned paths, and optimized trajectories.

        When *racing* is ``True`` only the adjusted (optimized) trajectories
        and the start/goal markers are rendered, giving a clean view of the
        routes the vehicles are tracking.

        Args:
            rrt_revealed: Number of RRT* tree nodes to display (ignored when
                *racing* is ``True``).
            sst_revealed: Number of SST tree nodes to display (ignored when
                *racing* is ``True``).
            racing: When ``True``, collapse the view to only the adjusted
                trajectories and start/goal markers.
        """
        x_min, x_max = self._bounds[0]
        y_min, y_max = self._bounds[1]

        if not racing:
            # Full world: SDF heatmap, obstacles, barriers, exploration trees,
            # raw planned paths, and optimized trajectories.
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
            if rrt_revealed >= self.rrt_total and self._rrt_path is not None:
                rrt_path_alpha = _PATH_ALPHA if self._rrt_traj_states else 1.0
                renderer_gl.draw_path(
                    self._rrt_path,
                    *_c(_C_RRT_PATH),
                    width=1.5 if self._rrt_traj_states else 2.5,
                    alpha=rrt_path_alpha,
                )
                if self._rrt_traj_states:
                    renderer_gl.draw_path(
                        self._rrt_traj_states, *_c(_C_RRT_TRAJ), width=3.0
                    )

            renderer_gl.draw_tree(
                self._sst_nodes,
                self._sst_parent,
                sst_revealed,
                *_c(_C_SST_EDGE),
                *_c(_C_SST_NODE),
            )
            if sst_revealed >= self.sst_total and self._sst_path is not None:
                sst_path_alpha = _PATH_ALPHA if self._sst_traj_states else 1.0
                renderer_gl.draw_path(
                    self._sst_path,
                    *_c(_C_SST_PATH),
                    width=1.5 if self._sst_traj_states else 2.5,
                    alpha=sst_path_alpha,
                )
                if self._sst_traj_states:
                    renderer_gl.draw_path(
                        self._sst_traj_states, *_c(_C_SST_TRAJ), width=3.0
                    )
        else:
            # Racing: full backdrop (road dots + buildings) so the obstacle
            # field stays visible, then the adjusted trajectories.
            renderer_gl.draw_obstacle_points(
                self._road_dots, *_c(_C_ROAD_DOT), point_size=3.0
            )
            renderer_gl.draw_obstacle_points(
                self._occ.points, *_c(_C_BUILDING), point_size=5.0
            )
            if self._rrt_traj_states:
                renderer_gl.draw_path(
                    self._rrt_traj_states, *_c(_C_RRT_TRAJ), width=3.0
                )
            elif self._rrt_path is not None:
                renderer_gl.draw_path(
                    self._rrt_path, *_c(_C_RRT_PATH), width=2.5
                )
            if self._sst_traj_states:
                renderer_gl.draw_path(
                    self._sst_traj_states, *_c(_C_SST_TRAJ), width=3.0
                )
            elif self._sst_path is not None:
                renderer_gl.draw_path(
                    self._sst_path, *_c(_C_SST_PATH), width=2.5
                )

        # Start/goal markers — always visible.
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


# Backward-compatible alias kept during refactor.
SparseScene = CityScene


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_occupancy(
    mesh_positions: list[tuple[float, float]],
    mesh_edges: list[tuple[int, int]],
    obs_cfg: dict[str, Any],
    cfg: dict[str, Any],
) -> Any:
    """Build obstacle occupancy from the triangular neighborhood mesh.

    Samples a regular grid over the world, retains only points that are
    (a) inside the Delaunay convex hull of the mesh nodes and
    (b) further than ``road_half_width`` from every road-edge centerline.
    The retained points form the KDTree obstacle cloud representing city
    blocks.  Road corridors (within ``road_half_width`` of any edge) remain
    obstacle-free.

    Args:
        mesh_positions: Node coordinates from
            :func:`_generate_neighborhood_mesh`.
        mesh_edges: Edge index pairs from
            :func:`_generate_neighborhood_mesh`.
        obs_cfg: Obstacle-layout dict providing ``world_width``,
            ``world_height``, ``road_half_width``, and
            ``obstacle_sampling_spacing``.
        cfg: Planner configuration dict providing ``obstacle_clearance``.

    Returns:
        A :class:`~arco.mapping.KDTreeOccupancy` ready for collision queries.
    """
    from arco.mapping import KDTreeOccupancy

    road_hw = float(obs_cfg["road_half_width"])
    spacing = float(obs_cfg["obstacle_sampling_spacing"])
    w = float(obs_cfg["world_width"])
    h = float(obs_cfg["world_height"])

    # Reconstruct Delaunay tessellation for inside-mesh test.
    pts_arr = np.array(mesh_positions)
    tri = _Delaunay(pts_arr)

    # Precompute road-edge segment data as arrays for vectorised distance.
    # seg_data[k] = (x1, y1, dx, dy, L2) for edge k.
    seg_data: list[tuple[float, float, float, float, float]] = []
    for u, v in mesh_edges:
        x1, y1 = mesh_positions[u]
        x2, y2 = mesh_positions[v]
        dx, dy = x2 - x1, y2 - y1
        seg_data.append((x1, y1, dx, dy, dx * dx + dy * dy))

    # Candidate sample grid.
    xs = np.arange(spacing / 2.0, w, spacing)
    ys = np.arange(spacing / 2.0, h, spacing)
    xx, yy = np.meshgrid(xs, ys)
    candidates = np.column_stack([xx.ravel(), yy.ravel()])

    # Discard samples outside the mesh convex hull.
    inside = tri.find_simplex(candidates) >= 0
    candidates = candidates[inside]

    # Discard samples within road_half_width of any road edge.
    mask = np.ones(len(candidates), dtype=bool)
    cx = candidates[:, 0]
    cy_arr = candidates[:, 1]
    for x1, y1, dx, dy, L2 in seg_data:
        if L2 < 1e-10:
            dist = np.sqrt((cx - x1) ** 2 + (cy_arr - y1) ** 2)
        else:
            vx = cx - x1
            vy = cy_arr - y1
            t = np.clip((vx * dx + vy * dy) / L2, 0.0, 1.0)
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            dist = np.sqrt((cx - proj_x) ** 2 + (cy_arr - proj_y) ** 2)
        mask &= dist > road_hw

    all_pts = candidates[mask].tolist()
    return KDTreeOccupancy(all_pts, clearance=float(cfg["obstacle_clearance"]))
