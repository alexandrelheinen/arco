#!/usr/bin/env python
"""RRT* vs SST race in a 3-D warehouse PPP robot environment.

Uses **PyOpenGL** (``pygame.OPENGL | pygame.DOUBLEBUF``) for hardware-
accelerated 3-D rendering — the first step toward a Gazebo-style frontend.
OpenGL provides depth testing, Phong lighting, and proper 3-D geometry
without any software painter's-algorithm hacks.

Both planners race from a start corner to the opposite goal corner of a
60 m x 20 m x 6 m warehouse bay. Three width-crossing barriers (tall,
small, then split-half with mixed heights) increase the difficulty of the
3-D route selection. Exploration trees are *not* shown — only the final
paths are rendered.

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

import logging
import math

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
    glMultMatrixf,
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

from arco.config import load_config
from arco.control import JointSpaceTracker
from arco.simulator.scenes.ppp import BOUNDS as _SCENE_BOUNDS
from arco.simulator.scenes.ppp import PPPScene
from arco.simulator.scenes.ppp import is_wall as _is_wall_box
from arco.simulator.sim.loading import run_with_loading_screen
from arco.simulator.sim.video import VideoWriter

logger = logging.getLogger(__name__)


def _perspective_matrix(
    fovy_deg: float, aspect: float, z_near: float, z_far: float
) -> np.ndarray:
    """Build a perspective projection matrix in OpenGL column-major order.

    Equivalent to ``gluPerspective`` without requiring GLU.

    Args:
        fovy_deg: Vertical field-of-view in degrees.
        aspect: Viewport width / height.
        z_near: Near clipping plane distance.
        z_far: Far clipping plane distance.

    Returns:
        16-element float32 array in column-major order for ``glMultMatrixf``.
    """
    f = 1.0 / math.tan(math.radians(fovy_deg) * 0.5)
    dz = z_near - z_far
    return np.array(
        [
            f / aspect,
            0.0,
            0.0,
            0.0,
            0.0,
            f,
            0.0,
            0.0,
            0.0,
            0.0,
            (z_far + z_near) / dz,
            -1.0,
            0.0,
            0.0,
            2.0 * z_far * z_near / dz,
            0.0,
        ],
        dtype=np.float32,
    )


def _look_at_matrix(
    eye: list[float], center: list[float], up: list[float]
) -> np.ndarray:
    """Build a look-at view matrix in OpenGL column-major order.

    Equivalent to ``gluLookAt`` without requiring GLU.

    Args:
        eye: Camera position (x, y, z).
        center: Look-at target (x, y, z).
        up: World up vector (x, y, z).

    Returns:
        16-element float32 array in column-major order for ``glMultMatrixf``.
    """
    e = np.array(eye, dtype=np.float64)
    c = np.array(center, dtype=np.float64)
    u = np.array(up, dtype=np.float64)
    f = c - e
    f /= np.linalg.norm(f)
    s = np.cross(f, u)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    return np.array(
        [
            s[0],
            u[0],
            -f[0],
            0.0,
            s[1],
            u[1],
            -f[1],
            0.0,
            s[2],
            u[2],
            -f[2],
            0.0,
            -np.dot(s, e),
            -np.dot(u, e),
            np.dot(f, e),
            1.0,
        ],
        dtype=np.float32,
    )


_DEFAULT_SCREEN_W = 1280
_DEFAULT_SCREEN_H = 800

_HOLD_SECS: float = 2.0
_POST_FINISH_SECS: float = 3.0

# Camera defaults
_CAM_AZIM: float = math.radians(40)
_CAM_ELEV: float = math.radians(28)
_CAM_DIST: float = 72.0
_WS_CENTER: tuple[float, float, float] = (
    (_SCENE_BOUNDS[0][0] + _SCENE_BOUNDS[0][1]) / 2.0,
    (_SCENE_BOUNDS[1][0] + _SCENE_BOUNDS[1][1]) / 2.0,
    (_SCENE_BOUNDS[2][0] + _SCENE_BOUNDS[2][1]) / 4.0,
)
_CAM_ROT_SPEED: float = math.radians(50)
_CAM_AUTO_ROT: float = math.radians(7)  # slow orbit for recording
_CAM_ZOOM_STEP: float = 0.03

# Small positive epsilon used to guard against division by zero in
# arc-length interpolation (any segment shorter than this is treated
# as zero-length).
_EPSILON: float = 1e-3

# OpenGL colors (float 0-1)
_C_TRAIL_RRT = (0.60, 0.80, 1.00)  # brighter variant of _C_RRT — blue
_C_TRAIL_SST = (0.35, 0.85, 0.45)  # brighter variant of _C_SST — green
_C_LA_RRT = (0.90, 0.95, 1.00)
_C_LA_SST = (0.70, 1.00, 0.75)
_BG = (18 / 255, 22 / 255, 32 / 255, 1.0)
_C_WALL = (0.68, 0.31, 0.17)
_C_WALL_EDGE = (0.38, 0.17, 0.09)
_C_BOX = (0.56, 0.41, 0.23)
_C_BOX_EDGE = (0.31, 0.23, 0.13)
_C_GRID = (0.15, 0.17, 0.22)
_C_RRT = (0.05, 0.05, 0.25)  # raw RRT* path — dark blue
_C_RRT_PRUNED: tuple[float, float, float] = (0.55, 0.72, 1.00)  # accent blue
_C_SST = (0.05, 0.22, 0.08)  # raw SST path — dark green
_C_SST_PRUNED: tuple[float, float, float] = (0.45, 1.00, 0.60)  # accent green
_C_TRAJ_RRT: tuple[float, float, float] = (0.38, 0.52, 0.88)  # medium blue
_C_TRAJ_SST: tuple[float, float, float] = (0.18, 0.68, 0.38)  # medium green
_C_START = (0.22, 0.86, 0.33)
_C_GOAL = (0.86, 0.30, 0.86)

# HUD colors (pygame RGB int 0-255)
_HC_RRT = (51, 51, 102)  # dark blue
_HC_SST = (30, 100, 50)  # dark green
_HC_HUD = (220, 220, 220)
_HC_DIM = (120, 120, 130)
_HC_SHADOW = (25, 30, 42)
_HC_WINNER = (255, 215, 50)
_HC_TIE = (200, 200, 80)

# End-effector half-dimensions (meters)
_EFF_HXY: float = 0.25
_EFF_HZ: float = 0.40


def _format_clock(seconds: float) -> str:
    """Format seconds as ``MMminSSs`` rounded to whole seconds."""
    rounded = int(round(seconds))
    mins, secs = divmod(rounded, 60)
    return f"{mins:02d}min{secs:02d}s"


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
        dist: Distance from :attr:`center` in meters.
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
    """Initialize the OpenGL state machine for the warehouse scene.

    Sets up depth testing, Phong lighting with a single overhead
    directional light, color-material tracking, and the perspective
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
    glMultMatrixf(_perspective_matrix(45.0, sw / max(sh, 1), 0.1, 1000.0))
    glMatrixMode(GL_MODELVIEW)


def _set_camera(camera: Camera3D) -> None:
    """Apply the camera look-at to the current modelview matrix.

    Args:
        camera: The orbit camera to apply.
    """
    glLoadIdentity()
    ex, ey, ez = camera.eye
    cx, cy, cz = camera.center
    glMultMatrixf(_look_at_matrix([ex, ey, ez], [cx, cy, cz], [0.0, 0.0, 1.0]))


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

    Lighting is disabled while drawing so edge color is exact.

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
    """Draw a 3-D path as a colored GL_LINE_STRIP.

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


def _draw_waypoints_3d(
    path: list[np.ndarray], r: float, g: float, b: float
) -> None:
    """Draw pruned-path nodes as small flat squares (GL_QUADS, z-plane).

    Only the node positions are drawn — no connecting lines.  Each square
    lies in the XY plane at the waypoint's Z elevation, making them visible
    from any oblique viewing angle.

    Args:
        path: Ordered list of 3-D waypoints.
        r: Red channel in [0, 1].
        g: Green channel in [0, 1].
        b: Blue channel in [0, 1].
    """
    if not path:
        return
    h = 0.18
    glDisable(GL_LIGHTING)
    glColor3f(r, g, b)
    glBegin(GL_QUADS)
    for pt in path:
        x, y, z = float(pt[0]), float(pt[1]), float(pt[2])
        glVertex3f(x - h, y - h, z)
        glVertex3f(x + h, y - h, z)
        glVertex3f(x + h, y + h, z)
        glVertex3f(x - h, y + h, z)
    glEnd()
    glEnable(GL_LIGHTING)


def _draw_floor_grid(x_max: float, y_max: float, spacing: float = 2.0) -> None:
    """Draw a ground-plane reference grid for depth perception.

    Args:
        x_max: Grid x extent in meters.
        y_max: Grid y extent in meters.
        spacing: Cell size in meters.
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
        pos: End-effector base center (x, y, z).
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
    rrt_metrics: dict,
    sst_metrics: dict,
    phase: str,
    hold_remaining: float,
) -> pygame.Surface:
    """Build a small semi-transparent status panel surface.

    Args:
        font: Monospace font.
        rrt_metrics: RRT* metrics dictionary.
        sst_metrics: SST metrics dictionary.
        phase: Current phase string (``"show"``, ``"race"``, ``"done"``).
        hold_remaining: Seconds until race start (only shown in ``"show"``).

    Returns:
        Transparent RGBA pygame surface.
    """

    def _planner_lines(name: str, metrics: dict) -> list[str]:
        return [
            name,
            (
                "  Planner steps / nodes: "
                f"{int(metrics['steps'])} / {int(metrics['nodes'])}"
            ),
            (
                "  Planner time: "
                f"{_format_clock(float(metrics['planner_time']))}"
            ),
            (
                "  Planned path length: "
                f"{int(round(float(metrics['planned_path_length'])))} m"
            ),
            (
                "  Trajectory arc length: "
                f"{int(round(float(metrics['trajectory_arc_length'])))} m"
            ),
            (
                "  Predicted duration: "
                f"{_format_clock(float(metrics['trajectory_duration']))}"
            ),
            f"  Path status: {metrics['path_status']}",
            f"  Optimizer status: {metrics['optimizer_status']}",
        ]

    lines: list[tuple[str, tuple[int, int, int]]] = [
        *[(line, _HC_RRT) for line in _planner_lines("RRT*", rrt_metrics)],
        ("", _HC_HUD),
        *[(line, _HC_SST) for line in _planner_lines("SST", sst_metrics)],
    ]
    if phase == "show":
        lines.append((f"  race in {hold_remaining:.1f} s", _HC_DIM))

    lh = font.get_linesize() + 2
    surf_w = 536
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
    """Build a translucent centered winner-banner surface.

    Args:
        big_font: Large bold font.
        text: Banner text (e.g. ``"RRT* WINS!"``).
        color: Text color.

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


class PPPRobot:
    """PPP gantry robot with independent per-axis velocity/acceleration limits.

    Each prismatic joint (x, y, z) is driven independently toward a moving
    carrot target on the planned path using a proportional position-to-velocity
    controller.  Desired velocity is proportional to the distance to the
    carrot, so the end-effector decelerates naturally as it approaches each
    carrot position and never overshoots.  An acceleration clamp limits how
    fast the velocity command can change, producing smooth transitions.

    Args:
        start: Initial 3-D end-effector position.
        max_vel: Maximum speed per axis in m/s.
        max_acc: Maximum acceleration per axis in m/s².
        proportional_gain: P-gain mapping distance error to velocity (s⁻¹).
            ``desired_vel = proportional_gain * err`` clipped to *max_vel*.
    """

    def __init__(
        self,
        start: np.ndarray,
        max_vel: float,
        max_acc: float,
        proportional_gain: float = 2.0,
    ) -> None:
        self.pos: np.ndarray = start.astype(float).copy()
        self.vel: np.ndarray = np.zeros(3)
        self._max_vel = max_vel
        self._max_acc = max_acc
        self._k_p = proportional_gain

    def step(self, target: np.ndarray, dt: float) -> None:
        """Advance one timestep toward *target* with per-axis limits.

        Computes a proportional velocity command directed at *target*
        (``v = k_p * err``, clipped to ``max_vel``) and applies a
        per-axis acceleration clamp so each joint's velocity changes are
        limited to :attr:`_max_acc` * *dt* per step independently.
        Because the commanded velocity scales down as the robot nears the
        carrot, there is no overshoot.

        Args:
            target: 3-D carrot position on the planned path.
            dt: Integration timestep in seconds.
        """
        err = target - self.pos
        # Proportional control: v_desired ∝ error → natural deceleration near
        # carrot, zero overshoot.  Clip each axis independently at max_vel.
        desired_vel = np.clip(self._k_p * err, -self._max_vel, self._max_vel)
        # Per-axis acceleration limit: each joint is rate-limited independently.
        dv = desired_vel - self.vel
        max_dv = self._max_acc * dt
        dv = np.clip(dv, -max_dv, max_dv)
        self.vel = np.clip(self.vel + dv, -self._max_vel, self._max_vel)
        self.pos = self.pos + self.vel * dt


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
    h = 0.18  # half-extent in meters
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    _draw_box(x - h, y - h, z - h, x + h, y + h, z + h, r, g, b)
    _draw_box_edges(x - h, y - h, z - h, x + h, y + h, z + h, r, g, b)


# ---------------------------------------------------------------------------
# Race simulation
# ---------------------------------------------------------------------------


def run_race(
    scene: PPPScene,
    cfg: dict,
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
    the ``race_speed`` configured in ``ppp.yml``.

    Phase 3 — **post-finish**: once both robots reach the goal, the
    simulation continues for :data:`_POST_FINISH_SECS` seconds before
    exiting.

    Recording requires a real or virtual OpenGL-capable display (use
    ``xvfb-run -a`` on headless Linux systems).

    Args:
        scene: Fully built :class:`~scenes.ppp.PPPScene`.
        cfg: Configuration dict loaded from ``tools/config/ppp.yml``.
        fps: Target frame rate in frames per second.
        dt: Simulation timestep in seconds.
        record: Output MP4 path.  Empty string = interactive mode.
        record_duration: Maximum recording length in seconds.
    """
    recording = bool(record)
    max_frames = int(fps * record_duration)

    sim_cfg = cfg.get("simulator", cfg)
    race_speed = float(sim_cfg["race_speed"])
    max_joint_vel = float(sim_cfg["max_joint_vel"])
    max_joint_acc = float(sim_cfg["max_joint_acc"])
    max_carrot_lag = float(sim_cfg["max_carrot_lag"])
    goal_reach_dist = float(sim_cfg["goal_reach_dist"])
    proportional_gain = float(sim_cfg.get("proportional_gain", 2.0))
    repulsion_gain = float(sim_cfg.get("repulsion_gain", 0.0))

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
    run_with_loading_screen(scene, sw, sh, bg_color=(18, 22, 32))
    pygame.display.set_caption(scene.title)

    font = pygame.font.SysFont("monospace", 14)
    big_font = pygame.font.SysFont("monospace", 40, bold=True)
    hint_surf = _hint_surface(font)

    camera = Camera3D()
    rrt_path = scene.rrt_path  # pruned waypoints
    sst_path = scene.sst_path  # pruned waypoints
    rrt_raw_path = scene.rrt_raw_path  # dense pre-pruning path
    sst_raw_path = scene.sst_raw_path
    # Use optimized trajectory (if available) as the carrot path so the
    # robot follows the time-optimal route; fall back to raw plan.
    rrt_nav = scene.rrt_traj if scene.rrt_traj else rrt_path
    sst_nav = scene.sst_traj if scene.sst_traj else sst_path
    boxes = scene.boxes
    rrt_metrics = scene.rrt_metrics
    sst_metrics = scene.sst_metrics

    rrt_arcs = _arc_lengths(rrt_nav) if rrt_nav else [0.0]
    sst_arcs = _arc_lengths(sst_nav) if sst_nav else [0.0]

    # Race state
    hold_timer = 0.0
    _max_vel_3d = np.full(3, max_joint_vel)
    _max_acc_3d = np.full(3, max_joint_acc)
    rrt_robot = JointSpaceTracker(
        max_vel=_max_vel_3d,
        max_acc=_max_acc_3d,
        proportional_gain=proportional_gain,
        occupancy=scene.occ,
        repulsion_gain=repulsion_gain,
    )
    rrt_robot.reset(scene.start)
    sst_robot = JointSpaceTracker(
        max_vel=_max_vel_3d,
        max_acc=_max_acc_3d,
        proportional_gain=proportional_gain,
        occupancy=scene.occ,
        repulsion_gain=repulsion_gain,
    )
    sst_robot.reset(scene.start)
    rrt_carrot_dist = 0.0
    sst_carrot_dist = 0.0
    rrt_carrot = scene.start.copy()
    sst_carrot = scene.start.copy()
    # Actual end-effector position histories (not the planned path).
    rrt_trail: list[np.ndarray] = [scene.start.copy()]
    sst_trail: list[np.ndarray] = [scene.start.copy()]
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
                        hold_timer = 0.0
                        rrt_robot.reset(scene.start)
                        sst_robot.reset(scene.start)
                        rrt_carrot_dist = 0.0
                        sst_carrot_dist = 0.0
                        rrt_carrot = scene.start.copy()
                        sst_carrot = scene.start.copy()
                        rrt_trail = [scene.start.copy()]
                        sst_trail = [scene.start.copy()]
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
                    if rrt_nav and not rrt_done:
                        rrt_lag = float(
                            np.linalg.norm(rrt_robot.q - rrt_carrot)
                        )
                        if rrt_lag < max_carrot_lag:
                            rrt_carrot_dist = min(
                                rrt_carrot_dist + race_speed * dt,
                                rrt_arcs[-1],
                            )
                        rrt_carrot, _ = _path_pos(
                            rrt_nav, rrt_arcs, rrt_carrot_dist
                        )
                        rrt_robot.step(rrt_carrot, dt)
                        rrt_trail.append(rrt_robot.q.copy())
                        rrt_done = (
                            float(np.linalg.norm(rrt_robot.q - scene.goal))
                            < goal_reach_dist
                        )
                    if sst_nav and not sst_done:
                        sst_lag = float(
                            np.linalg.norm(sst_robot.q - sst_carrot)
                        )
                        if sst_lag < max_carrot_lag:
                            sst_carrot_dist = min(
                                sst_carrot_dist + race_speed * dt,
                                sst_arcs[-1],
                            )
                        sst_carrot, _ = _path_pos(
                            sst_nav, sst_arcs, sst_carrot_dist
                        )
                        sst_robot.step(sst_carrot, dt)
                        sst_trail.append(sst_robot.q.copy())
                        sst_done = (
                            float(np.linalg.norm(sst_robot.q - scene.goal))
                            < goal_reach_dist
                        )
                    if not winner:
                        if rrt_done and sst_done:
                            winner = "TIE!"
                        elif rrt_done:
                            winner = "RRT* WINS!"
                        elif sst_done:
                            winner = "SST WINS!"
                    if rrt_done and sst_done:
                        phase = "done"
                elif phase == "done":
                    post_timer += dt
                    if recording and post_timer >= _POST_FINISH_SECS:
                        running = False

            # --- 3-D render (OpenGL) ------------------------------------
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            _set_camera(camera)

            _draw_floor_grid(scene.bounds[0][1], scene.bounds[1][1])

            for box in boxes:
                if _is_wall_box(box):
                    fc, ec = _C_WALL, _C_WALL_EDGE
                else:
                    fc, ec = _C_BOX, _C_BOX_EDGE
                _draw_box(*box, *fc)
                _draw_box_edges(*box, *ec)

            if rrt_raw_path:
                # Raw path (dense, pre-pruning) — dim polyline.
                _draw_path(rrt_raw_path, *_C_RRT)
            if rrt_path:
                # Pruned waypoints — accent squares (nodes only, no edges).
                _draw_waypoints_3d(rrt_path, *_C_RRT_PRUNED)
            if scene.rrt_traj:
                _draw_path(scene.rrt_traj, *_C_TRAJ_RRT)
            if sst_raw_path:
                _draw_path(sst_raw_path, *_C_SST)
            if sst_path:
                _draw_waypoints_3d(sst_path, *_C_SST_PRUNED)
            if scene.sst_traj:
                _draw_path(scene.sst_traj, *_C_TRAJ_SST)

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
                # Carrot markers — the target the robot is chasing.
                if phase == "race":
                    if rrt_path and not rrt_done:
                        _draw_lookahead_3d(rrt_carrot, *_C_LA_RRT)
                    if sst_path and not sst_done:
                        _draw_lookahead_3d(sst_carrot, *_C_LA_SST)
                _draw_effector(rrt_robot.q, *_C_RRT)
                _draw_effector(sst_robot.q, *_C_SST)

            # --- 2-D HUD overlay ----------------------------------------
            _blit_overlay(
                _status_surface(
                    font,
                    rrt_metrics,
                    sst_metrics,
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


def main(cfg: dict, save_path: str | None, sim_duration: float) -> None:
    sim_cfg = load_config("simulator")

    scene = PPPScene(cfg)
    run_race(
        scene,
        cfg,
        fps=sim_cfg["fps"],
        record=save_path,
        record_duration=sim_duration,
    )
