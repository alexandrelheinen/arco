#!/usr/bin/env python
"""RRT* vs SST race in a 3-D warehouse PPP robot environment.

Uses **PyOpenGL** (``pygame.OPENGL | pygame.DOUBLEBUF``) for hardware-
accelerated 3-D rendering — the first step toward a Gazebo-style frontend.
OpenGL provides depth testing, Phong lighting, and proper 3-D geometry
without any software painter's-algorithm hacks.

Both planners race from a start corner to the opposite goal corner of a
20 m x 10 m x 6 m warehouse bay.  A full-width blocking wall forces both
paths to arc above z = 4.5 m.  Exploration trees are *not* shown — only
the final paths are rendered.

Camera controls
---------------
LEFT / RIGHT   Rotate azimuth
UP / DOWN      Change elevation
+  /  -        Zoom in / out

General controls
----------------
SPACE          Pause / resume
R              Restart
Q / Escape     Quit

Usage
-----
::

    cd tools/simulator
    python main/ppp.py

Record a video (requires a real or virtual display with OpenGL support)::

    xvfb-run -a python main/ppp.py --record /tmp/ppp.mp4 --record-duration 90
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_HERE, ".."))

import numpy as np
import pygame
from OpenGL.GL import (  # type: ignore[import-untyped]
    GL_AMBIENT,
    GL_AMBIENT_AND_DIFFUSE,
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_COLOR_MATERIAL,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_DIFFUSE,
    GL_FRONT_AND_BACK,
    GL_LESS,
    GL_LIGHT0,
    GL_LIGHTING,
    GL_LINE_STRIP,
    GL_LINES,
    GL_MODELVIEW,
    GL_NEAREST,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_POLYGON_OFFSET_FILL,
    GL_POSITION,
    GL_PROJECTION,
    GL_QUADS,
    GL_RGBA,
    GL_SMOOTH,
    GL_SRC_ALPHA,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_UNSIGNED_BYTE,
    glBegin,
    glBindTexture,
    glBlendFunc,
    glClear,
    glClearColor,
    glColor3f,
    glColor4f,
    glColorMaterial,
    glDeleteTextures,
    glDepthFunc,
    glDisable,
    glEnable,
    glEnd,
    glGenTextures,
    glLightfv,
    glLineWidth,
    glLoadIdentity,
    glMatrixMode,
    glNormal3f,
    glOrtho,
    glPolygonOffset,
    glPopMatrix,
    glPushMatrix,
    glShadeModel,
    glTexCoord2f,
    glTexImage2D,
    glTexParameteri,
    glVertex2f,
    glVertex3f,
)
from OpenGL.GLU import (  # type: ignore[import-untyped]
    gluLookAt,
    gluPerspective,
)
from scenes.ppp import PPPScene
from scenes.ppp import is_wall as _is_wall_box
from sim.video import VideoWriter

from config import load_config

logger = logging.getLogger(__name__)

_DEFAULT_SCREEN_W = 1280
_DEFAULT_SCREEN_H = 800

_HOLD_SECS: float = 2.0
_POST_FINISH_SECS: float = 2.5
_RACE_SPEED: float = 1.5  # m/s along arc length
_LOOKAHEAD_DIST: float = 3.0  # metres ahead for lookahead marker

# Camera defaults
_CAM_AZIM: float = math.radians(40)
_CAM_ELEV: float = math.radians(28)
_CAM_DIST: float = 36.0
_WS_CENTER: tuple[float, float, float] = (10.0, 5.0, 1.5)
_CAM_ROT_SPEED: float = math.radians(50)
_CAM_AUTO_ROT: float = math.radians(7)  # slow orbit for recording
_CAM_ZOOM_STEP: float = 0.03

# Small positive epsilon used to guard against division by zero in
# arc-length interpolation (any segment shorter than this is treated
# as zero-length).
_EPSILON: float = 1e-9

# OpenGL colours (float 0-1)
_C_TRAIL_RRT = (0.60, 0.80, 1.00)  # brighter variant of _C_RRT
_C_TRAIL_SST = (0.50, 1.00, 0.88)  # brighter variant of _C_SST
_C_LA_RRT = (0.90, 0.95, 1.00)
_C_LA_SST = (0.80, 1.00, 0.95)
_BG = (18 / 255, 22 / 255, 32 / 255, 1.0)
_C_WALL = (0.68, 0.31, 0.17)
_C_WALL_EDGE = (0.38, 0.17, 0.09)
_C_BOX = (0.56, 0.41, 0.23)
_C_BOX_EDGE = (0.31, 0.23, 0.13)
_C_GRID = (0.15, 0.17, 0.22)
_C_RRT = (0.40, 0.64, 1.00)
_C_SST = (0.23, 0.90, 0.75)
_C_START = (0.22, 0.86, 0.33)
_C_GOAL = (0.86, 0.30, 0.86)

# HUD colours (pygame RGB int 0-255)
_HC_RRT = (102, 163, 255)
_HC_SST = (60, 229, 191)
_HC_HUD = (220, 220, 220)
_HC_DIM = (120, 120, 130)
_HC_SHADOW = (25, 30, 42)
_HC_WINNER = (255, 215, 50)
_HC_TIE = (200, 200, 80)

# End-effector half-dimensions (metres)
_EFF_HXY: float = 0.25
_EFF_HZ: float = 0.40


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


class Camera3D:
    """Orbit camera parameterised by azimuth, elevation, and distance.

    The camera orbits around :attr:`center` on a sphere of radius
    :attr:`dist`.  Azimuth rotates around the world Z-axis (z-up
    convention); elevation tilts the view above the horizontal plane.

    Args:
        azim: Initial azimuth in radians.
        elev: Initial elevation in radians.
        dist: Distance from :attr:`center` in metres.
        center: World-space point the camera looks at.
    """

    def __init__(
        self,
        azim: float = _CAM_AZIM,
        elev: float = _CAM_ELEV,
        dist: float = _CAM_DIST,
        center: tuple[float, float, float] = _WS_CENTER,
    ) -> None:
        self.azim = azim
        self.elev = min(math.radians(80), max(math.radians(5), elev))
        self.dist = max(2.0, dist)
        self.center = center

    @property
    def eye(self) -> tuple[float, float, float]:
        """Camera eye position in world coordinates."""
        cx, cy, cz = self.center
        ex = cx + self.dist * math.cos(self.elev) * math.sin(self.azim)
        ey = cy - self.dist * math.cos(self.elev) * math.cos(self.azim)
        ez = cz + self.dist * math.sin(self.elev)
        return ex, ey, ez


# ---------------------------------------------------------------------------
# OpenGL initialisation
# ---------------------------------------------------------------------------


def _gl_init(sw: int, sh: int) -> None:
    """Initialise the OpenGL state machine for the warehouse scene.

    Sets up depth testing, Phong lighting with a single overhead
    directional light, colour-material tracking, and the perspective
    projection matrix.

    Args:
        sw: Viewport width in pixels.
        sh: Viewport height in pixels.
    """
    glClearColor(*_BG)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)
    glShadeModel(GL_SMOOTH)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

    # Directional light from upper-right-front (w=0 = directional).
    glLightfv(GL_LIGHT0, GL_POSITION, [0.6, -0.4, 1.0, 0.0])
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.32, 0.32, 0.32, 1.0])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.78, 0.78, 0.78, 1.0])

    # Polygon offset so wireframe edges sit cleanly on top of filled faces.
    glPolygonOffset(1.0, 1.0)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, sw / max(sh, 1), 0.1, 1000.0)
    glMatrixMode(GL_MODELVIEW)


def _set_camera(camera: Camera3D) -> None:
    """Apply the camera look-at to the current modelview matrix.

    Args:
        camera: The orbit camera to apply.
    """
    glLoadIdentity()
    ex, ey, ez = camera.eye
    cx, cy, cz = camera.center
    gluLookAt(ex, ey, ez, cx, cy, cz, 0.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 3-D drawing primitives
# ---------------------------------------------------------------------------


def _draw_box(
    x1: float,
    y1: float,
    z1: float,
    x2: float,
    y2: float,
    z2: float,
    r: float,
    g: float,
    b: float,
) -> None:
    """Draw a solid lit box using GL_QUADS with per-face normals.

    Args:
        x1: Minimum x.
        y1: Minimum y.
        z1: Minimum z.
        x2: Maximum x.
        y2: Maximum y.
        z2: Maximum z.
        r: Red channel in [0, 1].
        g: Green channel in [0, 1].
        b: Blue channel in [0, 1].
    """
    glColor3f(r, g, b)
    glEnable(GL_POLYGON_OFFSET_FILL)
    glBegin(GL_QUADS)
    # Top (+z)
    glNormal3f(0.0, 0.0, 1.0)
    glVertex3f(x1, y1, z2)
    glVertex3f(x2, y1, z2)
    glVertex3f(x2, y2, z2)
    glVertex3f(x1, y2, z2)
    # Bottom (-z)
    glNormal3f(0.0, 0.0, -1.0)
    glVertex3f(x1, y2, z1)
    glVertex3f(x2, y2, z1)
    glVertex3f(x2, y1, z1)
    glVertex3f(x1, y1, z1)
    # Front (-y)
    glNormal3f(0.0, -1.0, 0.0)
    glVertex3f(x1, y1, z1)
    glVertex3f(x2, y1, z1)
    glVertex3f(x2, y1, z2)
    glVertex3f(x1, y1, z2)
    # Back (+y)
    glNormal3f(0.0, 1.0, 0.0)
    glVertex3f(x2, y2, z1)
    glVertex3f(x1, y2, z1)
    glVertex3f(x1, y2, z2)
    glVertex3f(x2, y2, z2)
    # Right (+x)
    glNormal3f(1.0, 0.0, 0.0)
    glVertex3f(x2, y1, z1)
    glVertex3f(x2, y2, z1)
    glVertex3f(x2, y2, z2)
    glVertex3f(x2, y1, z2)
    # Left (-x)
    glNormal3f(-1.0, 0.0, 0.0)
    glVertex3f(x1, y2, z1)
    glVertex3f(x1, y1, z1)
    glVertex3f(x1, y1, z2)
    glVertex3f(x1, y2, z2)
    glEnd()
    glDisable(GL_POLYGON_OFFSET_FILL)


def _draw_box_edges(
    x1: float,
    y1: float,
    z1: float,
    x2: float,
    y2: float,
    z2: float,
    r: float,
    g: float,
    b: float,
) -> None:
    """Draw the 12 wireframe edges of a box using GL_LINES.

    Lighting is disabled while drawing so edge colour is exact.

    Args:
        x1: Minimum x.
        y1: Minimum y.
        z1: Minimum z.
        x2: Maximum x.
        y2: Maximum y.
        z2: Maximum z.
        r: Red channel in [0, 1].
        g: Green channel in [0, 1].
        b: Blue channel in [0, 1].
    """
    glDisable(GL_LIGHTING)
    glColor3f(r, g, b)
    glBegin(GL_LINES)
    # Bottom ring
    glVertex3f(x1, y1, z1)
    glVertex3f(x2, y1, z1)
    glVertex3f(x2, y1, z1)
    glVertex3f(x2, y2, z1)
    glVertex3f(x2, y2, z1)
    glVertex3f(x1, y2, z1)
    glVertex3f(x1, y2, z1)
    glVertex3f(x1, y1, z1)
    # Top ring
    glVertex3f(x1, y1, z2)
    glVertex3f(x2, y1, z2)
    glVertex3f(x2, y1, z2)
    glVertex3f(x2, y2, z2)
    glVertex3f(x2, y2, z2)
    glVertex3f(x1, y2, z2)
    glVertex3f(x1, y2, z2)
    glVertex3f(x1, y1, z2)
    # Verticals
    glVertex3f(x1, y1, z1)
    glVertex3f(x1, y1, z2)
    glVertex3f(x2, y1, z1)
    glVertex3f(x2, y1, z2)
    glVertex3f(x2, y2, z1)
    glVertex3f(x2, y2, z2)
    glVertex3f(x1, y2, z1)
    glVertex3f(x1, y2, z2)
    glEnd()
    glEnable(GL_LIGHTING)


def _draw_path(path: list[np.ndarray], r: float, g: float, b: float) -> None:
    """Draw a 3-D path as a coloured GL_LINE_STRIP.

    Args:
        path: Ordered list of 3-D waypoints.
        r: Red channel in [0, 1].
        g: Green channel in [0, 1].
        b: Blue channel in [0, 1].
    """
    if len(path) < 2:
        return
    glDisable(GL_LIGHTING)
    glLineWidth(3.0)
    glColor3f(r, g, b)
    glBegin(GL_LINE_STRIP)
    for pt in path:
        glVertex3f(float(pt[0]), float(pt[1]), float(pt[2]))
    glEnd()
    glLineWidth(1.0)
    glEnable(GL_LIGHTING)


def _draw_floor_grid(x_max: float, y_max: float, spacing: float = 2.0) -> None:
    """Draw a ground-plane reference grid for depth perception.

    Args:
        x_max: Grid x extent in metres.
        y_max: Grid y extent in metres.
        spacing: Cell size in metres.
    """
    glDisable(GL_LIGHTING)
    glColor3f(*_C_GRID)
    glBegin(GL_LINES)
    x = 0.0
    while x <= x_max + 1e-6:
        glVertex3f(x, 0.0, 0.0)
        glVertex3f(x, y_max, 0.0)
        x += spacing
    y = 0.0
    while y <= y_max + 1e-6:
        glVertex3f(0.0, y, 0.0)
        glVertex3f(x_max, y, 0.0)
        y += spacing
    glEnd()
    glEnable(GL_LIGHTING)


def _draw_effector(pos: np.ndarray, r: float, g: float, b: float) -> None:
    """Draw the PPP end-effector as a small lit parallelepiped with edges.

    Args:
        pos: End-effector base centre (x, y, z).
        r: Red channel in [0, 1].
        g: Green channel in [0, 1].
        b: Blue channel in [0, 1].
    """
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    _draw_box(
        x - _EFF_HXY,
        y - _EFF_HXY,
        z,
        x + _EFF_HXY,
        y + _EFF_HXY,
        z + 2.0 * _EFF_HZ,
        r,
        g,
        b,
    )
    _draw_box_edges(
        x - _EFF_HXY,
        y - _EFF_HXY,
        z,
        x + _EFF_HXY,
        y + _EFF_HXY,
        z + 2.0 * _EFF_HZ,
        0.9,
        0.9,
        0.9,
    )


# ---------------------------------------------------------------------------
# 2-D HUD overlay  (pygame surface -> OpenGL texture)
# ---------------------------------------------------------------------------


def _blit_overlay(
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
    projection remaps (px, py) from pygame top-left to OpenGL bottom-left.

    Args:
        src: Source pygame surface (must support RGBA export).
        px: Left edge in screen pixels (pygame convention, y from top).
        py: Top edge in screen pixels.
        sw: Total screen width in pixels.
        sh: Total screen height in pixels.
    """
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

    # Convert pygame top-left py to OpenGL bottom-left gl_y.
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


def _status_surface(
    font: pygame.font.Font,
    rrt_pct: int,
    sst_pct: int,
    rrt_found: bool,
    sst_found: bool,
    phase: str,
    hold_remaining: float,
) -> pygame.Surface:
    """Build a small semi-transparent status panel surface.

    Args:
        font: Monospace font.
        rrt_pct: RRT* completion percentage (0-100).
        sst_pct: SST completion percentage (0-100).
        rrt_found: Whether RRT* found a path.
        sst_found: Whether SST found a path.
        phase: Current phase string (``"show"``, ``"race"``, ``"done"``).
        hold_remaining: Seconds until race start (only shown in ``"show"``).

    Returns:
        Transparent RGBA pygame surface.
    """
    rrt_tag = "path found" if rrt_found else "no path"
    sst_tag = "path found" if sst_found else "no path"
    lines: list[tuple[str, tuple[int, int, int]]] = [
        ("RRT*", _HC_RRT),
        (f"  {rrt_tag:>12}  {rrt_pct:3d}%", _HC_RRT),
        ("", _HC_HUD),
        ("SST", _HC_SST),
        (f"  {sst_tag:>12}  {sst_pct:3d}%", _HC_SST),
    ]
    if phase == "show":
        lines.append((f"  race in {hold_remaining:.1f} s", _HC_DIM))

    lh = font.get_linesize() + 2
    surf_w = 268
    surf_h = len(lines) * lh + 10
    surf = pygame.Surface((surf_w, surf_h), pygame.SRCALPHA)
    surf.fill((10, 14, 24, 185))
    y = 5
    for text, color in lines:
        if not text:
            y += lh
            continue
        shadow = font.render(text, True, _HC_SHADOW)
        surf.blit(shadow, (9, y + 1))
        surf.blit(font.render(text, True, color), (8, y))
        y += lh
    return surf


def _hint_surface(font: pygame.font.Font) -> pygame.Surface:
    """Build a one-line keyboard hint surface.

    Args:
        font: Monospace font.

    Returns:
        Transparent RGBA pygame surface.
    """
    text = "← → ↑ ↓  rotate   +/-  zoom   SPACE  pause   R  restart"
    rendered = font.render(text, True, _HC_DIM)
    w, h = rendered.get_size()
    surf = pygame.Surface((w + 4, h + 4), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))
    surf.blit(font.render(text, True, _HC_SHADOW), (3, 3))
    surf.blit(rendered, (2, 2))
    return surf


def _banner_surface(
    big_font: pygame.font.Font,
    text: str,
    color: tuple[int, int, int],
) -> pygame.Surface:
    """Build a translucent centred winner-banner surface.

    Args:
        big_font: Large bold font.
        text: Banner text (e.g. ``"RRT* WINS!"``).
        color: Text colour.

    Returns:
        RGBA banner surface.
    """
    rendered = big_font.render(text, True, color)
    rw, rh = rendered.get_width(), rendered.get_height()
    pad = 20
    surf = pygame.Surface((rw + 2 * pad, rh + 2 * pad), pygame.SRCALPHA)
    surf.fill((8, 10, 22, 215))
    surf.blit(big_font.render(text, True, _HC_SHADOW), (pad + 1, pad + 1))
    surf.blit(rendered, (pad, pad))
    return surf


# ---------------------------------------------------------------------------
# Arc-length helpers
# ---------------------------------------------------------------------------


def _arc_lengths(path: list[np.ndarray]) -> list[float]:
    """Return cumulative arc lengths along *path* starting at 0.0.

    Args:
        path: Ordered list of 3-D waypoints.

    Returns:
        List of cumulative distances, same length as *path*.
    """
    arcs = [0.0]
    for i in range(len(path) - 1):
        arcs.append(arcs[-1] + float(np.linalg.norm(path[i + 1] - path[i])))
    return arcs


def _path_pos(
    path: list[np.ndarray],
    arcs: list[float],
    dist: float,
) -> tuple[np.ndarray, bool]:
    """Interpolate the position at arc-length *dist* along *path*.

    Args:
        path: Ordered list of waypoints.
        arcs: Precomputed cumulative arc lengths (from :func:`_arc_lengths`).
        dist: Arc-length distance from path start.

    Returns:
        ``(position, at_goal)`` — interpolated 3-D position and a flag
        that is ``True`` once the path end is reached.
    """
    if dist >= arcs[-1]:
        return path[-1].copy(), True
    for i in range(len(arcs) - 1):
        if arcs[i + 1] >= dist:
            seg = arcs[i + 1] - arcs[i]
            t = (dist - arcs[i]) / max(seg, _EPSILON)
            return path[i] + t * (path[i + 1] - path[i]), False
    return path[-1].copy(), True


def _path_trail(
    path: list[np.ndarray],
    arcs: list[float],
    dist: float,
) -> list[np.ndarray]:
    """Return the traveled portion of *path* up to arc-length *dist*.

    Args:
        path: Ordered list of 3-D waypoints.
        arcs: Precomputed cumulative arc lengths.
        dist: Current arc-length distance from the path start.

    Returns:
        Sub-list of waypoints from the start up to *dist*, with a linearly
        interpolated end point appended when *dist* falls inside a segment.
    """
    if dist <= 0.0 or len(path) < 2:
        return []
    trail: list[np.ndarray] = [path[0]]
    for i in range(1, len(path)):
        if arcs[i] <= dist:
            trail.append(path[i])
        else:
            seg = arcs[i] - arcs[i - 1]
            t = (dist - arcs[i - 1]) / max(seg, _EPSILON)
            trail.append(path[i - 1] + t * (path[i] - path[i - 1]))
            break
    return trail


def _path_lookahead(
    path: list[np.ndarray],
    arcs: list[float],
    dist: float,
    la_dist: float,
) -> np.ndarray:
    """Return the lookahead point *la_dist* metres ahead of *dist*.

    Args:
        path: Ordered list of 3-D waypoints.
        arcs: Precomputed cumulative arc lengths.
        dist: Current arc-length distance from the path start.
        la_dist: Look-ahead distance in metres.

    Returns:
        Interpolated 3-D position on the path at ``dist + la_dist``,
        clamped to the path end.
    """
    pos, _ = _path_pos(path, arcs, min(dist + la_dist, arcs[-1]))
    return pos


def _draw_lookahead_3d(pos: np.ndarray, r: float, g: float, b: float) -> None:
    """Draw a small glowing marker at a 3-D lookahead point.

    Renders a tiny lit box surrounded by bright unlit edges so it is
    clearly visible against both the path line and background geometry.

    Args:
        pos: Lookahead position (x, y, z).
        r: Red channel in [0, 1].
        g: Green channel in [0, 1].
        b: Blue channel in [0, 1].
    """
    h = 0.18  # half-extent in metres
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    _draw_box(x - h, y - h, z - h, x + h, y + h, z + h, r, g, b)
    _draw_box_edges(x - h, y - h, z - h, x + h, y + h, z + h, r, g, b)


# ---------------------------------------------------------------------------
# Race simulation
# ---------------------------------------------------------------------------


def run_race(
    scene: PPPScene,
    *,
    fps: int = 30,
    dt: float = 0.05,
    record: str = "",
    record_duration: float = 90.0,
) -> None:
    """Run the 3-D PPP warehouse race with OpenGL rendering.

    Phase 1 — **path reveal**: both planned paths are shown for
    :data:`_HOLD_SECS` seconds.

    Phase 2 — **race**: both end-effectors advance simultaneously at
    :data:`_RACE_SPEED` m/s; the first to reach the goal wins.

    Phase 3 — **winner**: a banner is shown for :data:`_POST_FINISH_SECS`
    seconds before the simulation exits.

    Recording requires a real or virtual OpenGL-capable display (use
    ``xvfb-run -a`` on headless Linux systems).

    Args:
        scene: Fully built :class:`~scenes.ppp.PPPScene`.
        fps: Target frame rate in frames per second.
        dt: Simulation timestep in seconds.
        record: Output MP4 path.  Empty string = interactive mode.
        record_duration: Maximum recording duration in seconds.
    """
    recording = bool(record)
    max_frames = int(fps * record_duration)

    pygame.init()
    sw, sh = _DEFAULT_SCREEN_W, _DEFAULT_SCREEN_H
    if not recording:
        info = pygame.display.Info()
        w = int(getattr(info, "current_w", 0) or 0)
        h = int(getattr(info, "current_h", 0) or 0)
        if w > 0 and h > 0:
            sw = max(640, int(w * 0.9))
            sh = max(480, int(h * 0.9))

    pygame.display.set_mode((sw, sh), pygame.OPENGL | pygame.DOUBLEBUF)
    _gl_init(sw, sh)

    # Build the scene (runs planners — may take several seconds).
    scene.build()
    pygame.display.set_caption(scene.title)

    font = pygame.font.SysFont("monospace", 14)
    big_font = pygame.font.SysFont("monospace", 40, bold=True)
    hint_surf = _hint_surface(font)

    camera = Camera3D()
    rrt_path = scene.rrt_path
    sst_path = scene.sst_path
    boxes = scene.boxes

    rrt_arcs = _arc_lengths(rrt_path) if rrt_path else [0.0]
    sst_arcs = _arc_lengths(sst_path) if sst_path else [0.0]

    # Race state
    hold_timer = 0.0
    rrt_dist = 0.0
    sst_dist = 0.0
    rrt_pos = scene.start.copy()
    sst_pos = scene.start.copy()
    rrt_trail: list[np.ndarray] = []
    sst_trail: list[np.ndarray] = []
    rrt_done = False
    sst_done = False
    winner: str = ""
    post_timer = 0.0
    phase: str = "show"
    paused = False
    frame_count = 0
    clock = pygame.time.Clock()

    video_writer: VideoWriter | None = None
    if recording and record:
        video_writer = VideoWriter(record, sw, sh, fps)
        video_writer.open()

    try:
        running = True
        while running:
            # --- Events -------------------------------------------------
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_r:
                        camera = Camera3D()
                        hold_timer = rrt_dist = sst_dist = 0.0
                        rrt_pos = scene.start.copy()
                        sst_pos = scene.start.copy()
                        rrt_trail = []
                        sst_trail = []
                        rrt_done = sst_done = False
                        winner = ""
                        post_timer = 0.0
                        phase = "show"
                        paused = False

            # --- Camera controls ----------------------------------------
            if not paused:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LEFT]:
                    camera.azim -= _CAM_ROT_SPEED * dt
                if keys[pygame.K_RIGHT]:
                    camera.azim += _CAM_ROT_SPEED * dt
                if keys[pygame.K_UP]:
                    camera.elev = min(
                        math.radians(80),
                        camera.elev + _CAM_ROT_SPEED * dt,
                    )
                if keys[pygame.K_DOWN]:
                    camera.elev = max(
                        math.radians(5),
                        camera.elev - _CAM_ROT_SPEED * dt,
                    )
                if keys[pygame.K_PLUS] or keys[pygame.K_EQUALS]:
                    camera.dist = max(
                        3.0, camera.dist * (1.0 - _CAM_ZOOM_STEP)
                    )
                if keys[pygame.K_MINUS]:
                    camera.dist *= 1.0 + _CAM_ZOOM_STEP
                if recording:
                    camera.azim += _CAM_AUTO_ROT * dt

            # --- Simulation step ----------------------------------------
            if not paused:
                if phase == "show":
                    hold_timer += dt
                    if hold_timer >= _HOLD_SECS:
                        phase = "race"
                elif phase == "race":
                    if rrt_path and not rrt_done:
                        rrt_dist += _RACE_SPEED * dt
                        rrt_pos, rrt_done = _path_pos(
                            rrt_path, rrt_arcs, rrt_dist
                        )
                        rrt_trail = _path_trail(rrt_path, rrt_arcs, rrt_dist)
                    if sst_path and not sst_done:
                        sst_dist += _RACE_SPEED * dt
                        sst_pos, sst_done = _path_pos(
                            sst_path, sst_arcs, sst_dist
                        )
                        sst_trail = _path_trail(sst_path, sst_arcs, sst_dist)
                    if not winner:
                        if rrt_done and sst_done:
                            winner = "TIE!"
                        elif rrt_done:
                            winner = "RRT* WINS!"
                        elif sst_done:
                            winner = "SST WINS!"
                    if winner:
                        phase = "done"
                elif phase == "done":
                    post_timer += dt
                    if post_timer >= _POST_FINISH_SECS:
                        running = False

            # --- 3-D render (OpenGL) ------------------------------------
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            _set_camera(camera)

            _draw_floor_grid(20.0, 10.0)

            for box in boxes:
                if _is_wall_box(box):
                    fc, ec = _C_WALL, _C_WALL_EDGE
                else:
                    fc, ec = _C_BOX, _C_BOX_EDGE
                _draw_box(*box, *fc)
                _draw_box_edges(*box, *ec)

            if rrt_path:
                _draw_path(rrt_path, *_C_RRT)
            if sst_path:
                _draw_path(sst_path, *_C_SST)

            # Trajectory tracebacks (traveled portions, drawn thicker).
            if phase in ("race", "done"):
                if len(rrt_trail) >= 2:
                    glDisable(GL_LIGHTING)
                    glLineWidth(5.0)
                    glColor3f(*_C_TRAIL_RRT)
                    glBegin(GL_LINE_STRIP)
                    for _pt in rrt_trail:
                        glVertex3f(float(_pt[0]), float(_pt[1]), float(_pt[2]))
                    glEnd()
                    glLineWidth(1.0)
                    glEnable(GL_LIGHTING)
                if len(sst_trail) >= 2:
                    glDisable(GL_LIGHTING)
                    glLineWidth(5.0)
                    glColor3f(*_C_TRAIL_SST)
                    glBegin(GL_LINE_STRIP)
                    for _pt in sst_trail:
                        glVertex3f(float(_pt[0]), float(_pt[1]), float(_pt[2]))
                    glEnd()
                    glLineWidth(1.0)
                    glEnable(GL_LIGHTING)

            for pad_pos, pad_col in (
                (scene.start, _C_START),
                (scene.goal, _C_GOAL),
            ):
                px, py, pz = (
                    float(pad_pos[0]),
                    float(pad_pos[1]),
                    float(pad_pos[2]),
                )
                _draw_box(
                    px - 0.4,
                    py - 0.4,
                    pz,
                    px + 0.4,
                    py + 0.4,
                    pz + 0.07,
                    *pad_col,
                )

            if phase in ("race", "done"):
                # Lookahead markers (only while racing, not after finish).
                if phase == "race":
                    if rrt_path and not rrt_done:
                        la_rrt = _path_lookahead(
                            rrt_path, rrt_arcs, rrt_dist, _LOOKAHEAD_DIST
                        )
                        _draw_lookahead_3d(la_rrt, *_C_LA_RRT)
                    if sst_path and not sst_done:
                        la_sst = _path_lookahead(
                            sst_path, sst_arcs, sst_dist, _LOOKAHEAD_DIST
                        )
                        _draw_lookahead_3d(la_sst, *_C_LA_SST)
                _draw_effector(rrt_pos, *_C_RRT)
                _draw_effector(sst_pos, *_C_SST)

            # --- 2-D HUD overlay ----------------------------------------
            rrt_pct = (
                min(100, int(100 * rrt_dist / max(rrt_arcs[-1], 1e-6)))
                if rrt_path
                else 0
            )
            sst_pct = (
                min(100, int(100 * sst_dist / max(sst_arcs[-1], 1e-6)))
                if sst_path
                else 0
            )
            _blit_overlay(
                _status_surface(
                    font,
                    rrt_pct,
                    sst_pct,
                    rrt_path is not None,
                    sst_path is not None,
                    phase,
                    max(0.0, _HOLD_SECS - hold_timer),
                ),
                8,
                8,
                sw,
                sh,
            )
            _blit_overlay(
                hint_surf,
                (sw - hint_surf.get_width()) // 2,
                sh - hint_surf.get_height() - 6,
                sw,
                sh,
            )
            if winner:
                w_color = _HC_TIE if winner == "TIE!" else _HC_WINNER
                ban = _banner_surface(big_font, winner, w_color)
                _blit_overlay(
                    ban,
                    (sw - ban.get_width()) // 2,
                    (sh - ban.get_height()) // 2,
                    sw,
                    sh,
                )

            pygame.display.flip()

            if video_writer is not None:
                video_writer.write_frame_gl()
                frame_count += 1
                if frame_count >= max_frames:
                    running = False
            else:
                clock.tick(fps)

    finally:
        if video_writer is not None:
            video_writer.close()
        pygame.quit()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and launch the PPP 3-D OpenGL race simulation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        metavar="N",
        help="Target frame rate (default: 30).",
    )
    parser.add_argument(
        "--record",
        metavar="PATH",
        default="",
        help=(
            "Record to this MP4 file.  Requires a real or virtual "
            "OpenGL display (use xvfb-run -a on headless Linux)."
        ),
    )
    parser.add_argument(
        "--record-duration",
        type=float,
        default=90.0,
        metavar="SECS",
        help="Maximum recording length in seconds (default: 90).",
    )
    args = parser.parse_args()

    cfg = load_config("ppp")
    scene = PPPScene(cfg)
    run_race(
        scene,
        fps=args.fps,
        record=args.record,
        record_duration=args.record_duration,
    )


if __name__ == "__main__":
    main()
