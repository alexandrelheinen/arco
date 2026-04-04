"""OpenGL 2-D rendering toolkit for the ARCO simulator.

All drawing functions operate in **world coordinates** via a ``glOrtho``
projection set up by :func:`setup_2d_projection` before any draw calls.

Lighting is disabled for all 2-D primitives.  Line widths are in pixels;
point sizes are in pixels.  Circle approximations use 24 segments.

The SDF background texture is baked from world-space coordinates directly —
no screen-to-world inverse transform is required.

Text overlays use pygame SRCALPHA surfaces uploaded as GL RGBA textures,
identical to the approach in ``main/ppp.py``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pygame
from OpenGL.GL import (  # type: ignore[import-untyped]
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_LIGHTING,
    GL_LINE_STRIP,
    GL_LINEAR,
    GL_LINES,
    GL_MODELVIEW,
    GL_NEAREST,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_POINTS,
    GL_PROJECTION,
    GL_QUADS,
    GL_REPLACE,
    GL_RGB,
    GL_RGBA,
    GL_SRC_ALPHA,
    GL_TEXTURE_2D,
    GL_TEXTURE_ENV,
    GL_TEXTURE_ENV_MODE,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TRIANGLE_FAN,
    GL_TRIANGLE_STRIP,
    GL_UNSIGNED_BYTE,
    glBegin,
    glBindTexture,
    glBlendFunc,
    glClear,
    glClearColor,
    glColor3f,
    glColor4f,
    glDeleteTextures,
    glDisable,
    glEnable,
    glEnd,
    glGenTextures,
    glLineWidth,
    glLoadIdentity,
    glMatrixMode,
    glOrtho,
    glPointSize,
    glPopMatrix,
    glPushMatrix,
    glTexCoord2f,
    glTexEnvf,
    glTexImage2D,
    glTexParameteri,
    glVertex2f,
)

_DISC_SEGS = 24


def check_trajectory_clearance(
    path,
    occ,
    safety_dist: float,
) -> tuple[bool, float]:
    """Return ``(is_safe, min_clearance)`` for *path* against *occ*.

    Iterates over all points in *path* and queries the occupancy map for the
    distance to the nearest obstacle.  A trajectory is considered safe when
    every point is at least *safety_dist* metres from the nearest obstacle.

    Args:
        path: Ordered sequence of waypoints.  Each element must support
            index access ``[0]``/``[1]`` for (x, y).
        occ: Occupancy map exposing
            ``nearest_obstacle(point) -> (float, np.ndarray)``.
        safety_dist: Minimum required clearance from any obstacle (metres).

    Returns:
        A tuple ``(is_safe, min_clearance)`` where *is_safe* is ``True`` when
        the entire trajectory satisfies the clearance requirement and
        *min_clearance* is the smallest observed obstacle distance.
    """
    min_d = float("inf")
    for p in path:
        d, _ = occ.nearest_obstacle(np.asarray(p[:2], dtype=float))
        if d < min_d:
            min_d = d
    return (min_d >= safety_dist, min_d)


def _c(t: tuple[int, int, int]) -> tuple[float, float, float]:
    """Convert an integer RGB tuple to normalised GL floats.

    Args:
        t: RGB tuple with values in ``[0, 255]``.

    Returns:
        Float RGB tuple with values in ``[0.0, 1.0]``.
    """
    return (t[0] / 255.0, t[1] / 255.0, t[2] / 255.0)


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def setup_2d_projection(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    sw: int,
    sh: int,
    margin_frac: float = 0.04,
) -> None:
    """Set up an orthographic 2-D projection in world coordinates.

    Adds a fractional margin around the world bounding box and expands the
    shorter axis so that the projection matches the screen aspect ratio.
    Sets ``GL_PROJECTION`` then restores ``GL_MODELVIEW`` as the active matrix.

    Args:
        x_min: World left edge in metres.
        x_max: World right edge in metres.
        y_min: World bottom edge in metres.
        y_max: World top edge in metres.
        sw: Screen width in pixels.
        sh: Screen height in pixels.
        margin_frac: Fraction of the world extent to add as uniform margin.
    """
    world_w = x_max - x_min
    world_h = y_max - y_min
    mx = world_w * margin_frac
    my = world_h * margin_frac

    screen_ar = sw / max(sh, 1)
    world_ar = (world_w + 2 * mx) / max(world_h + 2 * my, 1e-9)
    if world_ar < screen_ar:
        extra = ((world_h + 2 * my) * screen_ar - (world_w + 2 * mx)) / 2.0
        mx += extra
    else:
        extra = ((world_w + 2 * mx) / screen_ar - (world_h + 2 * my)) / 2.0
        my += extra

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(x_min - mx, x_max + mx, y_min - my, y_max + my, -1.0, 1.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def world_bounds_from_transform(
    tx: Any,
    sw: int,
    sh: int,
) -> tuple[float, float, float, float]:
    """Derive world-space window extents from a camera transform.

    Works with both ``WorldTransform`` (static full view) and
    ``FollowTransform`` (vehicle-following) by reading their ``_ox``,
    ``_oy``, and ``_scale`` attributes.

    Args:
        tx: Camera transform with ``_ox``, ``_oy``, and ``_scale``.
        sw: Screen width in pixels.
        sh: Screen height in pixels.

    Returns:
        ``(x_min, x_max, y_min, y_max)`` in world metres.
    """
    scale = tx._scale
    ox = tx._ox
    oy = tx._oy
    # sx = ox + wx * scale  →  wx = (sx - ox) / scale
    x_min = (0.0 - ox) / scale
    x_max = (float(sw) - ox) / scale
    # sy = oy - wy * scale  →  wy = (oy - sy) / scale
    y_min = (oy - float(sh)) / scale
    y_max = oy / scale
    return x_min, x_max, y_min, y_max


# ---------------------------------------------------------------------------
# SDF background
# ---------------------------------------------------------------------------


def bake_sdf_texture(
    occ: Any,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    bg_color: tuple[int, int, int],
    near_color: tuple[int, int, int],
    resolution: int = 200,
) -> int:
    """Bake the signed-distance field of *occ* into a GL texture.

    Samples a *resolution* × *resolution* grid in **world space**, queries
    obstacle distances in one batch, colour-maps the result, and uploads it
    as an RGB ``GL_TEXTURE_2D``.

    Args:
        occ: Occupancy object exposing ``query_distances(points)`` where
            ``points`` has shape ``(N, 2)`` and returns shape ``(N,)``.
        x_min: World left edge in metres.
        x_max: World right edge in metres.
        y_min: World bottom edge in metres.
        y_max: World top edge in metres.
        bg_color: RGB colour for far-from-obstacle regions.
        near_color: RGB colour for obstacle-adjacent regions.
        resolution: Grid side length.

    Returns:
        OpenGL texture ID (integer).
    """
    xs = np.linspace(x_min, x_max, resolution)
    ys = np.linspace(y_min, y_max, resolution)
    grid_x, grid_y = np.meshgrid(xs, ys)
    pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    distances = occ.query_distances(pts)

    vmax = float(np.percentile(distances, 80))
    if vmax > 0.0:
        t = np.clip(distances / vmax, 0.0, 1.0)
    else:
        t = np.ones_like(distances)

    t_img = t.reshape(resolution, resolution)

    r = (near_color[0] + t_img * (bg_color[0] - near_color[0])).astype(
        np.uint8
    )
    g = (near_color[1] + t_img * (bg_color[1] - near_color[1])).astype(
        np.uint8
    )
    b = (near_color[2] + t_img * (bg_color[2] - near_color[2])).astype(
        np.uint8
    )

    # Stack into (resolution, resolution, 3) row-major array.
    rgb = np.ascontiguousarray(np.stack([r, g, b], axis=2))

    tex_id = int(glGenTextures(1))
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGB,
        resolution,
        resolution,
        0,
        GL_RGB,
        GL_UNSIGNED_BYTE,
        rgb.tobytes(),
    )
    glBindTexture(GL_TEXTURE_2D, 0)
    return tex_id


def draw_sdf_background(
    tex_id: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> None:
    """Draw the SDF texture as a world-space quad.

    Args:
        tex_id: GL texture ID returned by :func:`bake_sdf_texture`.
        x_min: World left edge in metres.
        x_max: World right edge in metres.
        y_min: World bottom edge in metres.
        y_max: World top edge in metres.
    """
    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
    glColor4f(1.0, 1.0, 1.0, 1.0)
    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 0.0)
    glVertex2f(x_min, y_min)
    glTexCoord2f(1.0, 0.0)
    glVertex2f(x_max, y_min)
    glTexCoord2f(1.0, 1.0)
    glVertex2f(x_max, y_max)
    glTexCoord2f(0.0, 1.0)
    glVertex2f(x_min, y_max)
    glEnd()
    glDisable(GL_TEXTURE_2D)


# ---------------------------------------------------------------------------
# Obstacle points
# ---------------------------------------------------------------------------


def draw_obstacle_points(
    points: Any,
    r: float,
    g: float,
    b: float,
    point_size: float = 5.0,
) -> None:
    """Draw obstacle points as GL_POINTS in world coordinates.

    Args:
        points: Iterable of ``(x, y)`` world positions.
        r: Red component ``[0, 1]``.
        g: Green component ``[0, 1]``.
        b: Blue component ``[0, 1]``.
        point_size: Point diameter in pixels.
    """
    glDisable(GL_LIGHTING)
    glPointSize(point_size)
    glColor3f(r, g, b)
    glBegin(GL_POINTS)
    for pt in points:
        glVertex2f(float(pt[0]), float(pt[1]))
    glEnd()


# ---------------------------------------------------------------------------
# Exploration tree
# ---------------------------------------------------------------------------


def draw_tree(
    nodes: Sequence[Any],
    parent: dict[int, int | None],
    count: int,
    edge_r: float,
    edge_g: float,
    edge_b: float,
    node_r: float,
    node_g: float,
    node_b: float,
    node_size: float = 3.0,
) -> None:
    """Draw the first *count* exploration-tree nodes and their parent edges.

    Edges are drawn as ``GL_LINES``; nodes as ``GL_POINTS``.

    Args:
        nodes: Full list of tree nodes; each supports ``[0]`` (x) and ``[1]`` (y).
        parent: Mapping from node index to parent index, ``None`` for root.
        count: Number of nodes to draw.
        edge_r: Edge red component ``[0, 1]``.
        edge_g: Edge green component ``[0, 1]``.
        edge_b: Edge blue component ``[0, 1]``.
        node_r: Node red component ``[0, 1]``.
        node_g: Node green component ``[0, 1]``.
        node_b: Node blue component ``[0, 1]``.
        node_size: Node point size in pixels.
    """
    glDisable(GL_LIGHTING)
    n = min(count, len(nodes))
    if n == 0:
        return

    glLineWidth(1.0)
    glColor3f(edge_r, edge_g, edge_b)
    glBegin(GL_LINES)
    for i in range(n):
        p = parent.get(i)
        if p is not None:
            glVertex2f(float(nodes[p][0]), float(nodes[p][1]))
            glVertex2f(float(nodes[i][0]), float(nodes[i][1]))
    glEnd()

    glPointSize(node_size)
    glColor3f(node_r, node_g, node_b)
    glBegin(GL_POINTS)
    for i in range(n):
        glVertex2f(float(nodes[i][0]), float(nodes[i][1]))
    glEnd()


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def draw_path(
    path: Sequence[Any],
    r: float,
    g: float,
    b: float,
    width: float = 2.5,
) -> None:
    """Draw a planned path as a ``GL_LINE_STRIP``.

    Args:
        path: Ordered sequence of waypoints; each supports ``[0]`` (x) and ``[1]`` (y).
        r: Red component ``[0, 1]``.
        g: Green component ``[0, 1]``.
        b: Blue component ``[0, 1]``.
        width: Line width in pixels.
    """
    if len(path) < 2:
        return
    glDisable(GL_LIGHTING)
    glLineWidth(width)
    glColor3f(r, g, b)
    glBegin(GL_LINE_STRIP)
    for pt in path:
        glVertex2f(float(pt[0]), float(pt[1]))
    glEnd()


def draw_dashed_path(
    path: Sequence[Any],
    r: float,
    g: float,
    b: float,
    dash_len: float = 1.0,
    gap_len: float = 0.6,
) -> None:
    """Draw a dashed polyline by drawing only alternating world-space segments.

    Accumulates arc-length along the path and toggles between drawing and
    skipping every ``dash_len`` / ``gap_len`` metres.

    Args:
        path: Ordered sequence of waypoints; each supports ``[0]`` (x) and ``[1]`` (y).
        r: Red component ``[0, 1]``.
        g: Green component ``[0, 1]``.
        b: Blue component ``[0, 1]``.
        dash_len: Length of each drawn dash in world metres.
        gap_len: Length of each gap in world metres.
    """
    if len(path) < 2:
        return
    glDisable(GL_LIGHTING)
    glLineWidth(1.0)
    glColor3f(r, g, b)

    draw_seg = True
    accumulated = 0.0
    threshold = dash_len

    for i in range(len(path) - 1):
        x0, y0 = float(path[i][0]), float(path[i][1])
        x1, y1 = float(path[i + 1][0]), float(path[i + 1][1])
        seg_len = math.hypot(x1 - x0, y1 - y0)
        if seg_len == 0.0:
            continue

        remaining = seg_len
        sx, sy = x0, y0

        while remaining > 0.0:
            step = min(remaining, threshold - accumulated)
            frac = step / seg_len
            ex = sx + (x1 - x0) * frac
            ey = sy + (y1 - y0) * frac
            if draw_seg:
                glBegin(GL_LINES)
                glVertex2f(sx, sy)
                glVertex2f(ex, ey)
                glEnd()
            sx, sy = ex, ey
            accumulated += step
            remaining -= step
            if accumulated >= threshold:
                accumulated = 0.0
                draw_seg = not draw_seg
                threshold = dash_len if draw_seg else gap_len


# ---------------------------------------------------------------------------
# Road network
# ---------------------------------------------------------------------------


def draw_road_edge(
    pts: Sequence[Any],
    r: float,
    g: float,
    b: float,
    width: float = 1.0,
) -> None:
    """Draw a single road edge as a ``GL_LINE_STRIP``.

    Args:
        pts: Ordered sequence of ``(x, y)`` world waypoints.
        r: Red component ``[0, 1]``.
        g: Green component ``[0, 1]``.
        b: Blue component ``[0, 1]``.
        width: Line width in pixels.
    """
    if len(pts) < 2:
        return
    glDisable(GL_LIGHTING)
    glLineWidth(width)
    glColor3f(r, g, b)
    glBegin(GL_LINE_STRIP)
    for pt in pts:
        glVertex2f(float(pt[0]), float(pt[1]))
    glEnd()


# ---------------------------------------------------------------------------
# Disc / ring
# ---------------------------------------------------------------------------


def draw_disc(
    cx: float,
    cy: float,
    radius: float,
    r: float,
    g: float,
    b: float,
) -> None:
    """Draw a filled circle approximated as a 24-gon using ``GL_TRIANGLE_FAN``.

    Args:
        cx: Centre x in world metres.
        cy: Centre y in world metres.
        radius: Radius in world metres.
        r: Red component ``[0, 1]``.
        g: Green component ``[0, 1]``.
        b: Blue component ``[0, 1]``.
    """
    glDisable(GL_LIGHTING)
    glColor3f(r, g, b)
    glBegin(GL_TRIANGLE_FAN)
    glVertex2f(cx, cy)
    for i in range(_DISC_SEGS + 1):
        angle = 2.0 * math.pi * i / _DISC_SEGS
        glVertex2f(
            cx + radius * math.cos(angle), cy + radius * math.sin(angle)
        )
    glEnd()


def draw_ring(
    cx: float,
    cy: float,
    r_outer: float,
    r_inner: float,
    r: float,
    g: float,
    b: float,
    segments: int = 24,
) -> None:
    """Draw a filled annular ring using ``GL_TRIANGLE_STRIP``.

    Args:
        cx: Centre x in world metres.
        cy: Centre y in world metres.
        r_outer: Outer radius in world metres.
        r_inner: Inner radius in world metres.
        r: Red component ``[0, 1]``.
        g: Green component ``[0, 1]``.
        b: Blue component ``[0, 1]``.
        segments: Number of strip segments (higher = smoother).
    """
    glDisable(GL_LIGHTING)
    glColor3f(r, g, b)
    glBegin(GL_TRIANGLE_STRIP)
    for i in range(segments + 1):
        angle = 2.0 * math.pi * i / segments
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        glVertex2f(cx + r_outer * cos_a, cy + r_outer * sin_a)
        glVertex2f(cx + r_inner * cos_a, cy + r_inner * sin_a)
    glEnd()


# ---------------------------------------------------------------------------
# Oriented rectangle (vehicle)
# ---------------------------------------------------------------------------


def draw_oriented_rect(
    cx: float,
    cy: float,
    half_l: float,
    half_w: float,
    heading: float,
    r: float,
    g: float,
    b: float,
) -> None:
    """Draw a filled oriented rectangle in world coordinates.

    Args:
        cx: Centre x in world metres.
        cy: Centre y in world metres.
        half_l: Half-length (forward) in world metres.
        half_w: Half-width (lateral) in world metres.
        heading: Heading in radians (0 = east, π/2 = north).
        r: Red component ``[0, 1]``.
        g: Green component ``[0, 1]``.
        b: Blue component ``[0, 1]``.
    """
    glDisable(GL_LIGHTING)
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    corners = []
    for lx, ly in (
        (half_l, half_w),
        (half_l, -half_w),
        (-half_l, -half_w),
        (-half_l, half_w),
    ):
        wx = cx + lx * cos_h - ly * sin_h
        wy = cy + lx * sin_h + ly * cos_h
        corners.append((wx, wy))
    glColor3f(r, g, b)
    glBegin(GL_QUADS)
    for wx, wy in corners:
        glVertex2f(wx, wy)
    glEnd()


def draw_world_line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    r: float,
    g: float,
    b: float,
    width: float = 1.0,
) -> None:
    """Draw a single line segment in world coordinates.

    Args:
        x1: Start x in world metres.
        y1: Start y in world metres.
        x2: End x in world metres.
        y2: End y in world metres.
        r: Red component ``[0, 1]``.
        g: Green component ``[0, 1]``.
        b: Blue component ``[0, 1]``.
        width: Line width in pixels.
    """
    glDisable(GL_LIGHTING)
    glLineWidth(width)
    glColor3f(r, g, b)
    glBegin(GL_LINES)
    glVertex2f(x1, y1)
    glVertex2f(x2, y2)
    glEnd()


# ---------------------------------------------------------------------------
# Text / HUD overlay
# ---------------------------------------------------------------------------


def blit_overlay(
    src: pygame.Surface,
    px: int,
    py: int,
    sw: int,
    sh: int,
) -> None:
    """Upload *src* as an RGBA texture and draw it as a screen-aligned quad.

    ``pygame.image.tostring(..., flip=True)`` flips the rows so that byte
    row 0 in the texture corresponds to the bottom of the pygame surface,
    matching OpenGL's bottom-left framebuffer origin.  The 2-D ortho
    projection remaps ``(px, py)`` from pygame top-left to OpenGL bottom-left.

    Args:
        src: Source pygame surface (must support RGBA export).
        px: Left edge in screen pixels (pygame convention, y from top).
        py: Top edge in screen pixels.
        sw: Total screen width in pixels.
        sh: Total screen height in pixels.
    """
    from OpenGL.GL import (
        GL_BLEND,
        GL_DEPTH_TEST,
        GL_LIGHTING,
        GL_MODELVIEW,
        GL_NEAREST,
        GL_ONE_MINUS_SRC_ALPHA,
        GL_PROJECTION,
        GL_QUADS,
        GL_RGBA,
        GL_SRC_ALPHA,
        GL_TEXTURE_2D,
        GL_TEXTURE_MAG_FILTER,
        GL_TEXTURE_MIN_FILTER,
        GL_UNSIGNED_BYTE,
        glBegin,
        glBindTexture,
        glBlendFunc,
        glColor4f,
        glDeleteTextures,
        glDisable,
        glEnable,
        glEnd,
        glGenTextures,
        glLoadIdentity,
        glMatrixMode,
        glOrtho,
        glPopMatrix,
        glPushMatrix,
        glTexCoord2f,
        glTexImage2D,
        glTexParameteri,
        glVertex2f,
    )

    w, h = src.get_size()
    data = pygame.image.tostring(src, "RGBA", True)

    tex_id = int(glGenTextures(1))
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        w,
        h,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        data,
    )
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0.0, float(sw), 0.0, float(sh), -1.0, 1.0)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glEnable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)
    glColor4f(1.0, 1.0, 1.0, 1.0)

    gl_y = float(sh - py - h)
    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 0.0)
    glVertex2f(float(px), gl_y)
    glTexCoord2f(1.0, 0.0)
    glVertex2f(float(px + w), gl_y)
    glTexCoord2f(1.0, 1.0)
    glVertex2f(float(px + w), gl_y + float(h))
    glTexCoord2f(0.0, 1.0)
    glVertex2f(float(px), gl_y + float(h))
    glEnd()

    glDisable(GL_TEXTURE_2D)
    glDisable(GL_BLEND)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()

    glDeleteTextures(1, [tex_id])


# ---------------------------------------------------------------------------
# Legend colour square
# ---------------------------------------------------------------------------


def draw_color_square(
    px: int,
    py: int,
    size: int,
    r: float,
    g: float,
    b: float,
    sw: int,
    sh: int,
) -> None:
    """Draw a small filled square in screen space (for legend items).

    Temporarily switches to a screen-space ortho projection, draws the quad,
    then restores the world-space projection.

    Args:
        px: Left edge in screen pixels (pygame convention, y from top).
        py: Top edge in screen pixels.
        size: Square side length in pixels.
        r: Red component ``[0, 1]``.
        g: Green component ``[0, 1]``.
        b: Blue component ``[0, 1]``.
        sw: Screen width in pixels.
        sh: Screen height in pixels.
    """
    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)

    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0.0, float(sw), 0.0, float(sh), -1.0, 1.0)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    gl_y = float(sh - py - size)
    glColor3f(r, g, b)
    glBegin(GL_QUADS)
    glVertex2f(float(px), gl_y)
    glVertex2f(float(px + size), gl_y)
    glVertex2f(float(px + size), gl_y + float(size))
    glVertex2f(float(px), gl_y + float(size))
    glEnd()

    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()
