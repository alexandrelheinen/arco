"""A* planner scene for the ARCO unified simulator.

:class:`AStarScene` builds a procedural road network, plans an A* route
between the two farthest nodes, and exposes a smooth path for the vehicle-
tracking phase.  The background-reveal phase is skipped (``background_total``
is zero) because the road network and route are rendered as a static backdrop.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from arco.config.palette import annotation_rgb, layer_rgb, ui_rgb
from arco.simulator import renderer_gl
from arco.simulator.sim.scene import SimScene
from arco.simulator.sim.tracking import VehicleConfig

# Small metric offset applied to start/goal to exercise the projection logic.
_ENDPOINT_OFFSET_M = 1.0
_ACTIVATION_RADIUS = 12.0

# Color palette (int tuples → _c() converts to float)
_C_BG = ui_rgb("background")
_C_ROAD = (90, 90, 100)
_C_ROAD_ROUTE = layer_rgb("astar", "path")
_C_NODE = (70, 100, 130)
_C_NODE_START = annotation_rgb(dark_bg=True)
_C_NODE_GOAL = annotation_rgb(dark_bg=True)
_C_NODE_ROUTE = layer_rgb("astar", "path")
_C_SMOOTH_PATH = layer_rgb("astar", "trajectory")


def _c(t: tuple[int, int, int]) -> tuple[float, float, float]:
    return (t[0] / 255.0, t[1] / 255.0, t[2] / 255.0)


class AStarScene(SimScene):
    """A* route-planning scene over a procedural road network.

    The scene skips the background-reveal phase (``background_total == 0``)
    and goes straight to vehicle tracking. The full road network with the
    planned route and smooth path is rendered behind the vehicle at all times.

    Args:
        graph_config: Parsed graph configuration dict (from ``astar.yml`` ``graph``).
        vehicle_config: Parsed vehicle configuration dict (from
            ``astar.yml`` ``vehicle``).  The ``"dubins"`` sub-key is used.
    """

    def __init__(
        self,
        graph_config: dict[str, Any],
        vehicle_config: dict[str, Any],
    ) -> None:
        self._graph_cfg = graph_config
        self._veh_cfg = vehicle_config
        self._graph: Any = None
        self._route: list[int] = []
        self._smooth_path: list[tuple[float, float]] = []
        self._vehicle_config: VehicleConfig | None = None

    # ------------------------------------------------------------------
    # SimScene interface
    # ------------------------------------------------------------------

    def build(self, *, progress=None) -> None:  # type: ignore[override]
        """Build the road network and plan the A* route.

        Args:
            progress: Optional callable ``(step_name, step_index, total_steps)``
                invoked at each build milestone for loading-screen feedback.

        Raises:
            RuntimeError: If the route planner cannot connect start to goal.
        """
        _total = 2
        # Import here to avoid top-level side-effects.
        from arco.planning.discrete import RouteRouter
        from arco.simulator.graph.generator import generate_graph

        if progress is not None:
            progress("Generating road network", 1, _total)
        self._graph = generate_graph(self._graph_cfg)
        node_ids = list(self._graph.nodes)
        start_pos, goal_pos = _find_farthest_pair(self._graph, node_ids)

        start_position = np.array(
            [
                start_pos[0] + _ENDPOINT_OFFSET_M,
                start_pos[1] + _ENDPOINT_OFFSET_M,
            ]
        )
        goal_position = np.array(
            [
                goal_pos[0] - _ENDPOINT_OFFSET_M,
                goal_pos[1] - _ENDPOINT_OFFSET_M,
            ]
        )

        if progress is not None:
            progress("Planning A* route", 2, _total)
        router = RouteRouter(self._graph, activation_radius=_ACTIVATION_RADIUS)
        result = router.plan(start_position, goal_position)
        if result is None:
            raise RuntimeError(
                "A* route planning failed — check start/goal positions and "
                "activation_radius."
            )

        self._route = result.path
        self._smooth_path = _build_smooth_path(self._graph, result.path)

        dubins = self._veh_cfg["dubins"]
        self._vehicle_config = VehicleConfig(
            max_speed=float(dubins["max_speed"]),
            min_speed=0.0,
            cruise_speed=float(dubins["cruise_speed"]),
            lookahead_distance=float(dubins["lookahead"]),
            goal_radius=float(dubins["goal_radius"]),
            max_turn_rate=math.radians(float(dubins["max_turn_rate"])),
            max_acceleration=float(dubins["max_accel"]),
            max_turn_rate_dot=math.radians(float(dubins["max_turn_rate_dot"])),
            curvature_gain=float(dubins.get("curvature_gain", 0.0)),
        )

    @property
    def title(self) -> str:
        """Human-readable scene label."""
        graph_type = self._graph_cfg.get("type", "ring").title()
        return f"{graph_type} — A*"

    @property
    def bg_color(self) -> tuple[int, int, int]:
        """Background fill color."""
        return _C_BG

    @property
    def world_points(self) -> list[tuple[float, float]]:
        """All graph node positions for auto-fitting the full view."""
        return [
            (
                float(self._graph.position(n)[0]),
                float(self._graph.position(n)[1]),
            )
            for n in self._graph.nodes
        ]

    @property
    def zoom_world_points(self) -> list[tuple[float, float]]:
        """Route node positions for auto-fitting the zoomed view."""
        return [
            (
                float(self._graph.position(n)[0]),
                float(self._graph.position(n)[1]),
            )
            for n in self._route
        ]

    @property
    def waypoints(self) -> list[tuple[float, float]]:
        """Dense smooth path through route edge geometry."""
        return self._smooth_path

    @property
    def vehicle_config(self) -> VehicleConfig:
        """Vehicle and controller parameters."""
        assert self._vehicle_config is not None, "call build() first"
        return self._vehicle_config

    @property
    def background_total(self) -> int:
        """Zero — A* skips the background-reveal phase."""
        return 0

    def draw_background(self, revealed: int) -> None:
        """Draw the road network, route, and smooth path.

        Args:
            revealed: Ignored (always 0 for A*).
        """
        route_edge_set: set[tuple[int, int]] = set()
        if self._route:
            route_edge_set = {
                (min(a, b), max(a, b))
                for a, b in zip(self._route[:-1], self._route[1:])
            }

        for node_a, node_b, _ in self._graph.edges:
            pts = self._graph.full_edge_geometry(node_a, node_b)
            key = (min(node_a, node_b), max(node_a, node_b))
            if key in route_edge_set:
                renderer_gl.draw_road_edge(pts, *_c(_C_ROAD_ROUTE), width=3.0)
            else:
                renderer_gl.draw_road_edge(pts, *_c(_C_ROAD), width=1.0)

        route_set = set(self._route) if self._route else set()
        node_r = 0.3  # world meters
        for nid in self._graph.nodes:
            x, y = self._graph.position(nid)
            if self._route and nid == self._route[0]:
                renderer_gl.draw_disc(
                    float(x), float(y), node_r * 2, *_c(_C_NODE_START)
                )
            elif self._route and nid == self._route[-1]:
                renderer_gl.draw_disc(
                    float(x), float(y), node_r * 2, *_c(_C_NODE_GOAL)
                )
            elif nid in route_set:
                renderer_gl.draw_disc(
                    float(x), float(y), node_r * 1.4, *_c(_C_NODE_ROUTE)
                )
            else:
                renderer_gl.draw_disc(float(x), float(y), node_r, *_c(_C_NODE))

        if len(self._smooth_path) >= 2:
            renderer_gl.draw_dashed_path(
                self._smooth_path, *_c(_C_SMOOTH_PATH)
            )

    def sidebar_content(
        self, **state: Any
    ) -> list[tuple[list[str], tuple[int, int, int]]]:
        """Return sidebar lines (tracking phase only — no background reveal).

        Args:
            **state: Keys used: ``veh_step``, ``speed``, ``cte``, ``finished``.

        Returns:
            Single-element list with ``(lines, color)`` for the A* planner.
        """
        from arco.config.palette import layer_rgb

        color = layer_rgb("astar", "vehicle")
        veh_step = int(state.get("veh_step", 0))
        speed = float(state.get("speed", 0.0))
        cte = float(state.get("cte", 0.0))
        finished = bool(state.get("finished", False))
        lines: list[str] = [
            self.title,
            f"Step: {veh_step}",
            f"Speed: {speed:.1f} m/s",
            f"CTE: {cte:+.2f} m",
        ]
        if finished:
            lines.append("[ GOAL REACHED ]")
        return [(lines, color)]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _find_farthest_pair(
    graph: Any,
    node_ids: list[int],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return world positions of the two farthest-apart graph nodes.

    Args:
        graph: :class:`~arco.mapping.graph.road.RoadGraph` instance.
        node_ids: List of node identifiers to search over.

    Returns:
        Pair of ``(x, y)`` world positions.
    """
    best_dist = -1.0
    best_a = node_ids[0]
    best_b = node_ids[-1]
    for i in range(len(node_ids)):
        xi, yi = graph.position(node_ids[i])
        for j in range(i + 1, len(node_ids)):
            xj, yj = graph.position(node_ids[j])
            d = math.hypot(xi - xj, yi - yj)
            if d > best_dist:
                best_dist = d
                best_a = node_ids[i]
                best_b = node_ids[j]
    return (
        (float(graph.position(best_a)[0]), float(graph.position(best_a)[1])),
        (float(graph.position(best_b)[0]), float(graph.position(best_b)[1])),
    )


def _build_smooth_path(
    graph: Any,
    route: list[int],
) -> list[tuple[float, float]]:
    """Return a dense smooth path through all route edge waypoints.

    Args:
        graph: :class:`~arco.mapping.graph.road.RoadGraph` instance.
        route: Ordered list of node IDs defining the planned route.

    Returns:
        Ordered list of ``(x, y)`` waypoints.
    """
    if len(route) < 2:
        return (
            [
                (
                    float(graph.position(route[0])[0]),
                    float(graph.position(route[0])[1]),
                )
            ]
            if route
            else []
        )
    smooth: list[tuple[float, float]] = []
    for i in range(len(route) - 1):
        pts = graph.full_edge_geometry(route[i], route[i + 1])
        smooth.extend((float(p[0]), float(p[1])) for p in pts[:-1])
    last = graph.position(route[-1])
    smooth.append((float(last[0]), float(last[1])))
    return smooth
