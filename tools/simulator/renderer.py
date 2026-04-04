"""Pygame rendering adapter for the ARCO road network and vehicle simulation.

Draws the following layers (back to front):

- Road network edges as polylines with waypoint geometry
- Route edges highlighted in a distinct colour
- Smooth path as a dashed polyline
- Past vehicle trajectory as a fading blue trace
- Tracking lookahead target as a yellow circle
- Current vehicle as a green rectangle with heading arrow
- HUD text overlay (speed, cross-track error, step counter)
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import pygame

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C_BG = (28, 28, 35)
C_ROAD = (90, 90, 100)
C_ROAD_ROUTE = (220, 80, 60)
C_NODE = (70, 100, 130)
C_NODE_START = (60, 200, 90)
C_NODE_GOAL = (60, 100, 220)
C_NODE_ROUTE = (200, 100, 80)
C_SMOOTH_PATH = (230, 140, 40)
C_TRAJECTORY = (60, 140, 220)
C_TRACKING_TARGET = (240, 200, 0)
C_VEHICLE = (80, 220, 100)
C_VEHICLE_ARROW = (30, 255, 80)
C_HUD = (220, 220, 220)
C_HUD_SHADOW = (40, 40, 50)

# Vehicle display dimensions in *screen pixels*
_VEH_LENGTH_PX = 14
_VEH_WIDTH_PX = 7


class WorldTransform:
    """Maps world (metre) coordinates to pygame screen pixels.

    Preserves aspect ratio and adds a uniform margin.  World y-axis points
    **up**; screen y-axis points **down** — this class handles the flip.

    Args:
        nodes: Iterable of ``(x, y)`` world positions used to compute the
            bounding box automatically.
        screen_size: ``(width, height)`` of the pygame display in pixels.
        margin: Pixel margin added on every side.
    """

    def __init__(
        self,
        nodes: Sequence[tuple[float, float]],
        screen_size: tuple[int, int],
        margin: int = 60,
    ) -> None:
        """Initialise the transform from node positions and screen size."""
        xs = [n[0] for n in nodes]
        ys = [n[1] for n in nodes]
        wx_min, wx_max = min(xs), max(xs)
        wy_min, wy_max = min(ys), max(ys)

        w_avail = screen_size[0] - 2 * margin
        h_avail = screen_size[1] - 2 * margin
        scale_x = w_avail / max(wx_max - wx_min, 1.0)
        scale_y = h_avail / max(wy_max - wy_min, 1.0)
        self._scale = min(scale_x, scale_y)

        # Centring offsets so the bounding box is centred in the available area
        map_w_px = (wx_max - wx_min) * self._scale
        map_h_px = (wy_max - wy_min) * self._scale
        self._ox = margin + (w_avail - map_w_px) / 2.0 - wx_min * self._scale
        self._oy = (
            screen_size[1]
            - margin
            - (h_avail - map_h_px) / 2.0
            + wy_min * self._scale
        )

    def __call__(self, wx: float, wy: float) -> tuple[int, int]:
        """Convert world ``(x, y)`` to screen ``(col, row)`` integer pixels.

        Args:
            wx: World x-coordinate in metres.
            wy: World y-coordinate in metres.

        Returns:
            Integer ``(screen_x, screen_y)`` pixel coordinates.
        """
        sx = int(self._ox + wx * self._scale)
        sy = int(self._oy - wy * self._scale)
        return (sx, sy)

    @property
    def scale(self) -> float:
        """Pixels per metre."""
        return self._scale


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def draw_road_network(
    surface: pygame.Surface,
    graph,
    transform: WorldTransform,
    route: Sequence[int] | None = None,
) -> None:
    """Draw road edges and intersection nodes onto *surface*.

    Args:
        surface: Target pygame surface.
        graph: :class:`~arco.mapping.graph.road.RoadGraph` instance.
        transform: World-to-screen coordinate transform.
        route: Optional ordered sequence of node IDs defining the planned
            route.  Route edges are drawn in a highlight colour.
    """
    route_edge_set: set[tuple[int, int]] = set()
    if route:
        route_edge_set = {
            (min(a, b), max(a, b)) for a, b in zip(route[:-1], route[1:])
        }

    for node_a, node_b, _ in graph.edges:
        pts = graph.full_edge_geometry(node_a, node_b)
        screen_pts = [transform(float(p[0]), float(p[1])) for p in pts]
        key = (min(node_a, node_b), max(node_a, node_b))
        color = C_ROAD_ROUTE if key in route_edge_set else C_ROAD
        width = 3 if key in route_edge_set else 1
        if len(screen_pts) >= 2:
            pygame.draw.lines(surface, color, False, screen_pts, width)

    route_set = set(route) if route else set()
    for nid in graph.nodes:
        x, y = graph.position(nid)
        sx, sy = transform(float(x), float(y))
        if route and nid == route[0]:
            col, r = C_NODE_START, 6
        elif route and nid == route[-1]:
            col, r = C_NODE_GOAL, 6
        elif nid in route_set:
            col, r = C_NODE_ROUTE, 4
        else:
            col, r = C_NODE, 3
        pygame.draw.circle(surface, col, (sx, sy), r)


def draw_smooth_path(
    surface: pygame.Surface,
    smooth_path: Sequence[tuple[float, float]],
    transform: WorldTransform,
) -> None:
    """Draw the smooth interpolated path as a dashed orange polyline.

    Args:
        surface: Target pygame surface.
        smooth_path: Ordered sequence of ``(x, y)`` world waypoints.
        transform: World-to-screen coordinate transform.
    """
    if len(smooth_path) < 2:
        return
    pts = [transform(float(p[0]), float(p[1])) for p in smooth_path]
    dash_len = 6
    gap_len = 4
    draw_seg = True
    count = 0
    for i in range(len(pts) - 1):
        color = C_SMOOTH_PATH if draw_seg else None
        if color:
            pygame.draw.line(surface, color, pts[i], pts[i + 1], 1)
        count += 1
        if count >= (dash_len if draw_seg else gap_len):
            draw_seg = not draw_seg
            count = 0


def draw_trajectory(
    surface: pygame.Surface,
    trajectory: Sequence[tuple[float, float, float]],
    transform: WorldTransform,
) -> None:
    """Draw the vehicle's past trajectory as a fading blue polyline.

    Args:
        surface: Target pygame surface.
        trajectory: Ordered sequence of ``(x, y, heading)`` poses.
        transform: World-to-screen coordinate transform.
    """
    if len(trajectory) < 2:
        return
    pts = [transform(float(p[0]), float(p[1])) for p in trajectory]
    for i in range(1, len(pts)):
        pygame.draw.line(surface, C_TRAJECTORY, pts[i - 1], pts[i], 1)


def draw_tracking_target(
    surface: pygame.Surface,
    target: tuple[float, float],
    transform: WorldTransform,
) -> None:
    """Draw the lookahead tracking target as a yellow circle.

    Args:
        surface: Target pygame surface.
        target: World ``(x, y)`` of the lookahead target.
        transform: World-to-screen coordinate transform.
    """
    sx, sy = transform(float(target[0]), float(target[1]))
    pygame.draw.circle(surface, C_TRACKING_TARGET, (sx, sy), 6)
    pygame.draw.circle(surface, C_BG, (sx, sy), 3)


def draw_vehicle(
    surface: pygame.Surface,
    x: float,
    y: float,
    heading: float,
    transform: WorldTransform,
) -> None:
    """Draw the vehicle as a green oriented rectangle with a heading arrow.

    Args:
        surface: Target pygame surface.
        x: Vehicle x-position in world metres.
        y: Vehicle y-position in world metres.
        heading: Vehicle heading in radians (0 = east, π/2 = north).
        transform: World-to-screen coordinate transform.
    """
    cx, cy = transform(float(x), float(y))
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)

    half_l = _VEH_LENGTH_PX / 2
    half_w = _VEH_WIDTH_PX / 2

    # Four corners of the rectangle in local frame → rotated to world/screen
    # Screen y is inverted; forward direction is (cos_h, -sin_h) in screen space
    corners = []
    for lx, ly in [
        (half_l, half_w),
        (half_l, -half_w),
        (-half_l, -half_w),
        (-half_l, half_w),
    ]:
        # Rotate by heading (screen y inverted so sin negated)
        rx = lx * cos_h - ly * sin_h
        ry = -(lx * sin_h + ly * cos_h)
        corners.append((cx + rx, cy + ry))

    pygame.draw.polygon(surface, C_VEHICLE, corners)
    pygame.draw.polygon(surface, C_VEHICLE_ARROW, corners, 1)

    # Forward arrow
    arr_end = (
        cx + _VEH_LENGTH_PX * 0.7 * cos_h,
        cy - _VEH_LENGTH_PX * 0.7 * sin_h,
    )
    pygame.draw.line(surface, C_VEHICLE_ARROW, (cx, cy), arr_end, 2)


def draw_hud(
    surface: pygame.Surface,
    font: pygame.font.Font,
    step: int,
    speed: float,
    cross_track: float,
    finished: bool,
    paused: bool,
) -> None:
    """Draw heads-up display text in the top-left corner.

    Args:
        surface: Target pygame surface.
        font: Pygame font for rendering text.
        step: Current simulation step count.
        speed: Current vehicle speed in m/s.
        cross_track: Cross-track error in metres.
        finished: Whether the vehicle has reached the goal.
        paused: Whether the simulation is paused.
    """
    lines = [
        f"Step: {step}",
        f"Speed: {speed:.1f} m/s",
        f"CTE: {cross_track:+.1f} m",
    ]
    if paused:
        lines.append("  [ PAUSED — press SPACE ]")
    if finished:
        lines.append("  [ GOAL REACHED ]")

    x, y = 10, 10
    for line in lines:
        shadow = font.render(line, True, C_HUD_SHADOW)
        surface.blit(shadow, (x + 1, y + 1))
        text = font.render(line, True, C_HUD)
        surface.blit(text, (x, y))
        y += font.get_linesize() + 2


def draw_legend(surface: pygame.Surface, font: pygame.font.Font) -> None:
    """Draw a small colour legend in the bottom-left corner.

    Args:
        surface: Target pygame surface.
        font: Pygame font for rendering text.
    """
    items = [
        (C_ROAD, "Road"),
        (C_ROAD_ROUTE, "Route"),
        (C_SMOOTH_PATH, "Smooth path"),
        (C_TRAJECTORY, "Trajectory"),
        (C_NODE_START, "Start"),
        (C_NODE_GOAL, "Goal"),
        (C_TRACKING_TARGET, "Lookahead"),
        (C_VEHICLE, "Vehicle"),
    ]
    bh = surface.get_height()
    x, y = 10, bh - len(items) * (font.get_linesize() + 2) - 10
    for color, label in items:
        pygame.draw.rect(surface, color, (x, y + 3, 12, 10))
        text = font.render(label, True, C_HUD)
        surface.blit(text, (x + 16, y))
        y += font.get_linesize() + 2


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------


def _render_hud_lines(
    surface: pygame.Surface,
    font: pygame.font.Font,
    lines: list[str],
) -> None:
    """Blit *lines* as a shadowed HUD overlay in the top-left corner.

    Args:
        surface: Target pygame surface.
        font: Pygame font for rendering text.
        lines: Text lines to display, from top to bottom.
    """
    x, y = 10, 10
    for line in lines:
        shadow = font.render(line, True, C_HUD_SHADOW)
        surface.blit(shadow, (x + 1, y + 1))
        text = font.render(line, True, C_HUD)
        surface.blit(text, (x, y))
        y += font.get_linesize() + 2


# ---------------------------------------------------------------------------
# Shared primitives for sampling-based planner scenes
# ---------------------------------------------------------------------------


def draw_obstacles(
    surface: pygame.Surface,
    occ: Any,
    transform: Any,
    *,
    color: tuple[int, int, int] = (160, 60, 60),
) -> None:
    """Draw obstacle points from an occupancy map as small filled circles.

    Args:
        surface: Target pygame surface.
        occ: Occupancy object exposing an iterable ``points`` attribute.
        transform: Callable ``(wx, wy) -> (sx, sy)`` world-to-screen.
        color: RGB fill colour for each obstacle circle.
    """
    for pt in occ.points:
        sx, sy = transform(float(pt[0]), float(pt[1]))
        pygame.draw.circle(surface, color, (sx, sy), 4)


def draw_exploration_tree(
    surface: pygame.Surface,
    nodes: Sequence[Any],
    parent: dict[int, int | None],
    count: int,
    transform: Any,
    *,
    edge_color: tuple[int, int, int],
    node_color: tuple[int, int, int],
) -> None:
    """Draw the first *count* exploration-tree nodes and their parent edges.

    Args:
        surface: Target pygame surface.
        nodes: All tree nodes (full list from planner); each supports
            index access ``[0]`` and ``[1]`` for x and y.
        parent: Mapping from node index to parent index, ``None`` for root.
        count: Number of nodes to draw (starting from index 0).
        transform: Callable ``(wx, wy) -> (sx, sy)`` world-to-screen.
        edge_color: RGB colour for tree edges.
        node_color: RGB colour for tree node circles.
    """
    for i in range(min(count, len(nodes))):
        sx, sy = transform(float(nodes[i][0]), float(nodes[i][1]))
        p = parent.get(i)
        if p is not None:
            px, py = transform(float(nodes[p][0]), float(nodes[p][1]))
            pygame.draw.line(surface, edge_color, (px, py), (sx, sy), 1)
        pygame.draw.circle(surface, node_color, (sx, sy), 2)


def draw_planned_path(
    surface: pygame.Surface,
    path: Sequence[Any],
    transform: Any,
    *,
    color: tuple[int, int, int] = (230, 170, 30),
) -> None:
    """Draw the planner solution path as a thick polyline.

    Args:
        surface: Target pygame surface.
        path: Ordered sequence of waypoints; each supports index access
            ``[0]`` and ``[1]`` for x and y.
        transform: Callable ``(wx, wy) -> (sx, sy)`` world-to-screen.
        color: RGB colour for the path polyline.
    """
    if len(path) < 2:
        return
    pts = [transform(float(p[0]), float(p[1])) for p in path]
    pygame.draw.lines(surface, color, False, pts, 3)


def draw_endpoints(
    surface: pygame.Surface,
    start: Any,
    goal: Any,
    transform: Any,
    *,
    start_color: tuple[int, int, int],
    goal_color: tuple[int, int, int],
    bg_color: tuple[int, int, int],
) -> None:
    """Draw start and goal markers as annular circles.

    Args:
        surface: Target pygame surface.
        start: Start position; supports index access ``[0]`` and ``[1]``.
        goal: Goal position; supports index access ``[0]`` and ``[1]``.
        transform: Callable ``(wx, wy) -> (sx, sy)`` world-to-screen.
        start_color: Outer ring RGB colour for the start marker.
        goal_color: Outer ring RGB colour for the goal marker.
        bg_color: Inner fill RGB colour (same as background).
    """
    sx, sy = transform(float(start[0]), float(start[1]))
    gx, gy = transform(float(goal[0]), float(goal[1]))
    pygame.draw.circle(surface, start_color, (sx, sy), 8)
    pygame.draw.circle(surface, bg_color, (sx, sy), 4)
    pygame.draw.circle(surface, goal_color, (gx, gy), 8)
    pygame.draw.circle(surface, bg_color, (gx, gy), 4)


def draw_planning_hud(
    surface: pygame.Surface,
    font: pygame.font.Font,
    label: str,
    revealed: int,
    total: int,
    path_found: bool,
) -> None:
    """Draw a planning-phase HUD showing exploration progress.

    Args:
        surface: Target pygame surface.
        font: Pygame font for rendering text.
        label: Planner name shown at the top of the HUD (e.g. ``"RRT*"``).
        revealed: Number of tree nodes currently visible.
        total: Total number of tree nodes.
        path_found: Whether a solution path was found.
    """
    _render_hud_lines(
        surface,
        font,
        [
            label,
            f"Nodes: {revealed}/{total}",
            f"Path: {'found' if path_found else 'none'}",
        ],
    )


def draw_tracking_hud(
    surface: pygame.Surface,
    font: pygame.font.Font,
    label: str,
    step: int,
    speed: float,
    cte: float,
    finished: bool,
    paused: bool = False,
) -> None:
    """Draw a vehicle-tracking HUD showing controller state.

    Args:
        surface: Target pygame surface.
        font: Pygame font for rendering text.
        label: Scene label shown at the top of the HUD.
        step: Current simulation step count.
        speed: Current vehicle speed in m/s.
        cte: Cross-track error in metres.
        finished: Whether the vehicle has reached the goal.
        paused: Whether the simulation is currently paused.
    """
    lines = [
        f"{label} — tracking",
        f"Step: {step}",
        f"Speed: {speed:.1f} m/s",
        f"CTE: {cte:+.1f} m",
    ]
    if paused:
        lines.append("[ PAUSED — press SPACE ]")
    if finished:
        lines.append("[ GOAL REACHED ]")
    _render_hud_lines(surface, font, lines)


# ---------------------------------------------------------------------------
# SDF background baking
# ---------------------------------------------------------------------------


def bake_sdf_surface(
    occ: Any,
    transform: WorldTransform,
    screen_size: tuple[int, int],
    *,
    bg_color: tuple[int, int, int],
    near_color: tuple[int, int, int],
    resolution: int = 200,
) -> pygame.Surface:
    """Bake the signed-distance field of *occ* into a small pygame.Surface.

    Samples a *resolution* × *resolution* grid in **screen space**, converts
    each sample back to world coordinates via the inverse of *transform*, and
    queries obstacle distances in one batch.  Sampling in screen space
    guarantees pixel-accurate alignment with drawn obstacles regardless of
    the transform's margin, aspect-ratio correction, or centring offset.

    The returned surface should be smooth-scaled to the actual display size
    at draw time::

        scaled = pygame.transform.smoothscale(sdf_surface, screen_size)
        display_surface.blit(scaled, (0, 0))

    Args:
        occ: Occupancy object exposing ``query_distances(points)`` where
            ``points`` has shape ``(N, 2)`` and the return value has shape
            ``(N,)``.
        transform: ``WorldTransform`` used to draw the scene, providing the
            scale and offsets needed for the inverse mapping.
        screen_size: ``(width, height)`` of the pygame display in pixels.
        bg_color: RGB colour for far-from-obstacle regions.
        near_color: RGB colour for obstacle-adjacent regions.
        resolution: Grid side length in pixels before upscaling.

    Returns:
        A ``pygame.Surface`` of size ``(resolution, resolution)``.
    """
    import numpy as np

    sw, sh = screen_size
    col_samples = np.linspace(0.0, float(sw - 1), resolution)
    row_samples = np.linspace(0.0, float(sh - 1), resolution)
    cols, rows = np.meshgrid(col_samples, row_samples)

    # Inverse of WorldTransform: sx = ox + wx*scale, sy = oy - wy*scale
    wx = (cols - transform._ox) / transform._scale
    wy = (transform._oy - rows) / transform._scale

    grid_pts = np.stack([wx.ravel(), wy.ravel()], axis=1)
    distances = occ.query_distances(grid_pts)

    vmax = float(np.percentile(distances, 80))
    if vmax > 0.0:
        t = np.clip(distances / vmax, 0.0, 1.0)
    else:
        t = np.ones_like(distances)

    t_img = t.reshape(resolution, resolution)  # (rows=y, cols=x)

    # Lerp: t=0 → near_color, t=1 → bg_color
    r = (near_color[0] + t_img * (bg_color[0] - near_color[0])).astype(
        np.uint8
    )
    g = (near_color[1] + t_img * (bg_color[1] - near_color[1])).astype(
        np.uint8
    )
    b = (near_color[2] + t_img * (bg_color[2] - near_color[2])).astype(
        np.uint8
    )

    # pygame.surfarray.blit_array expects shape (width, height, 3) i.e.
    # (cols, rows, 3).  Stack channels as (rows, cols, 3) then transpose.
    rgb = np.stack([r, g, b], axis=2)
    rgb_hw = np.ascontiguousarray(rgb.transpose(1, 0, 2))  # (cols, rows, 3)

    surf = pygame.Surface((resolution, resolution))
    pygame.surfarray.blit_array(surf, rgb_hw)
    return surf
