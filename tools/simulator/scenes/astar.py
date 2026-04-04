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
import pygame
from renderer import draw_legend, draw_road_network, draw_smooth_path
from sim.scene import SimScene
from sim.tracking import VehicleConfig

# Small metric offset applied to start/goal to exercise the projection logic.
_ENDPOINT_OFFSET_M = 1.0
_ACTIVATION_RADIUS = 12.0


class AStarScene(SimScene):
    """A* route-planning scene over a procedural road network.

    The scene skips the background-reveal phase (``background_total == 0``)
    and goes straight to vehicle tracking. The full road network with the
    planned route and smooth path is rendered behind the vehicle at all times.

    Args:
        graph_config: Parsed graph configuration dict (from ``graph.yml``).
        vehicle_config: Parsed vehicle configuration dict (from
            ``vehicle.yml``).  The ``"dubins"`` sub-key is used.
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
        self._font: pygame.font.Font | None = None

    # ------------------------------------------------------------------
    # SimScene interface
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Build the road network and plan the A* route.

        Raises:
            RuntimeError: If the route planner cannot connect start to goal.
        """
        # Import here to avoid top-level side-effects.
        from graph.generator import generate_graph

        from arco.planning.discrete import RouteRouter

        self._graph = generate_graph(self._graph_cfg)
        node_ids = list(self._graph.nodes)
        start_pos, goal_pos = _find_farthest_pair(self._graph, node_ids)

        start_xy = np.array(
            [
                start_pos[0] + _ENDPOINT_OFFSET_M,
                start_pos[1] + _ENDPOINT_OFFSET_M,
            ]
        )
        goal_xy = np.array(
            [
                goal_pos[0] - _ENDPOINT_OFFSET_M,
                goal_pos[1] - _ENDPOINT_OFFSET_M,
            ]
        )

        router = RouteRouter(self._graph, activation_radius=_ACTIVATION_RADIUS)
        result = router.plan(start_xy, goal_xy)
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
        self._font = pygame.font.SysFont("monospace", 14)

    @property
    def title(self) -> str:
        """Human-readable scene label."""
        graph_type = self._graph_cfg.get("type", "ring").title()
        return f"{graph_type} — A*"

    @property
    def bg_color(self) -> tuple[int, int, int]:
        """Background fill colour."""
        return (28, 28, 35)

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

    def draw_background(
        self,
        surface: pygame.Surface,
        transform: object,
        revealed: int,
    ) -> None:
        """Draw the road network, route, and smooth path.

        Args:
            surface: Pygame surface to draw onto.
            transform: World-to-screen callable.
            revealed: Ignored (always 0 for A*).
        """
        draw_road_network(surface, self._graph, transform, self._route)
        draw_smooth_path(surface, self._smooth_path, transform)
        if self._font is not None:
            draw_legend(surface, self._font)

    def draw_background_hud(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        revealed: int,
    ) -> None:
        """No-op — never called because ``background_total`` is zero.

        Args:
            surface: Pygame surface to draw onto.
            font: Pygame font for rendering text.
            revealed: Ignored.
        """


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
