#!/usr/bin/env python
"""RRT* vs SST race in a 3-D RRP SCARA-like arm environment.

Uses **PyOpenGL** (``pygame.OPENGL | pygame.DOUBLEBUF``) for hardware-
accelerated 3-D rendering.

The RRP arm has two revolute joints (XY plane) and one vertical prismatic
joint.  The workspace is a cylindrical annulus with pillar obstacles that
force XY routing and slab barriers that force Z routing.

Both planners race from the start joint configuration to the goal in
3-D joint space ``(q1, q2, z)``.  The rendered view shows the 3-D
Cartesian arm pose (two links at the current Z height) inside the
cylindrical workspace.

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
    python main/rrp.py

Record a video (requires a real or virtual display with OpenGL support)::

    xvfb-run -a python main/rrp.py --record /tmp/rrp.mp4 --record-duration 90
"""

from __future__ import annotations

import argparse
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
    GL_LINE_LOOP,
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

from arco.control import JointSpaceTracker
from arco.tools.simulator.scenes.rrp import RRPScene
from arco.tools.simulator.sim.loading import run_with_loading_screen
from arco.tools.simulator.sim.video import VideoWriter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Matrix helpers
# ---------------------------------------------------------------------------


def _perspective_matrix(
    fovy_deg: float, aspect: float, z_near: float, z_far: float
) -> np.ndarray:
    """Build a perspective projection matrix in OpenGL column-major order."""
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
    """Build a look-at view matrix in OpenGL column-major order."""
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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_SCREEN_W = 1280
_DEFAULT_SCREEN_H = 800

_HOLD_SECS: float = 2.0
_POST_FINISH_SECS: float = 3.0

_CAM_AZIM: float = math.radians(50)
_CAM_ELEV: float = math.radians(35)
_CAM_DIST: float = 7.0
_WS_CENTER: tuple[float, float, float] = (0.0, 0.0, 2.0)
_CAM_ROT_SPEED: float = math.radians(50)
_CAM_AUTO_ROT: float = math.radians(7)
_CAM_ZOOM_STEP: float = 0.03

_EPSILON: float = 1e-3

# OpenGL float colours
_BG = (18 / 255, 22 / 255, 32 / 255, 1.0)
_C_PILLAR = (0.56, 0.41, 0.23)
_C_PILLAR_EDGE = (0.31, 0.23, 0.13)
_C_SLAB = (0.42, 0.55, 0.62)
_C_SLAB_EDGE = (0.22, 0.32, 0.38)
_C_GRID = (0.15, 0.17, 0.22)
_C_RRT = (0.05, 0.05, 0.25)
_C_SST = (0.05, 0.22, 0.08)
_C_TRAJ_RRT: tuple[float, float, float] = (0.38, 0.52, 0.88)  # medium blue
_C_TRAJ_SST: tuple[float, float, float] = (0.18, 0.68, 0.38)  # medium green
_C_TRAIL_RRT = (0.60, 0.80, 1.00)
_C_TRAIL_SST = (0.35, 0.85, 0.45)
_C_LA_RRT = (0.90, 0.95, 1.00)
_C_LA_SST = (0.70, 1.00, 0.75)
_C_START = (0.22, 0.86, 0.33)
_C_GOAL = (0.86, 0.30, 0.86)
_C_LINK1_RRT = (0.40, 0.55, 0.95)
_C_LINK2_RRT = (0.55, 0.70, 1.00)
_C_LINK1_SST = (0.25, 0.78, 0.40)
_C_LINK2_SST = (0.40, 0.92, 0.55)

# HUD colors (pygame RGB int 0-255)
_HC_RRT = (51, 51, 102)
_HC_SST = (30, 100, 50)
_HC_HUD = (220, 220, 220)
_HC_DIM = (120, 120, 130)
_HC_SHADOW = (25, 30, 42)
_HC_WINNER = (255, 215, 50)
_HC_TIE = (200, 200, 80)


# ---------------------------------------------------------------------------
# Pillar vs slab classification
# ---------------------------------------------------------------------------

# A pillar has a small XY footprint (both sides < this threshold) and full
# Z height.  Anything wider is treated as a slab.
_PILLAR_XY_THRESHOLD: float = 0.8


def _is_pillar(
    obs: list[float],
    z_max_full: float = 4.0,
    xy_thresh: float = _PILLAR_XY_THRESHOLD,
) -> bool:
    """Return ``True`` if *obs* is a pillar obstacle.

    Pillars are narrow in XY (both dimensions ≤ *xy_thresh*) and tall
    (Z range ≥ half the full workspace height).

    Args:
        obs: ``[x1, y1, z1, x2, y2, z2]``.
        z_max_full: Full workspace Z height in metres.
        xy_thresh: Maximum XY side length to be considered a pillar.
    """
    dx = obs[3] - obs[0]
    dy = obs[4] - obs[1]
    dz = obs[5] - obs[2]
    return dx <= xy_thresh and dy <= xy_thresh and dz >= z_max_full * 0.5


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


class Camera3D:
    """Orbit camera parameterised by azimuth, elevation, and distance.

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
        self.dist = max(0.5, dist)
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
    """Initialize the OpenGL state machine for the RRP arm scene.

    Args:
        sw: Viewport width in pixels.
        sh: Viewport height in pixels.
    """
    from OpenGL.GL import (  # already imported above but safe
        GL_DEPTH_BUFFER_BIT,
    )

    glClearColor(*_BG)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)
    glShadeModel(GL_SMOOTH)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

    glLightfv(GL_LIGHT0, GL_POSITION, [0.4, -0.6, 1.0, 0.0])
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.35, 0.35, 0.35, 1.0])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.75, 0.75, 0.75, 1.0])

    glPolygonOffset(1.0, 1.0)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glMultMatrixf(_perspective_matrix(45.0, sw / max(sh, 1), 0.01, 200.0))
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
        x1: Minimum x.  y1: Minimum y.  z1: Minimum z.
        x2: Maximum x.  y2: Maximum y.  z2: Maximum z.
        r, g, b: Face color channels in [0, 1].
    """
    glColor3f(r, g, b)
    glEnable(GL_POLYGON_OFFSET_FILL)
    glBegin(GL_QUADS)
    glNormal3f(0.0, 0.0, 1.0)
    glVertex3f(x1, y1, z2)
    glVertex3f(x2, y1, z2)
    glVertex3f(x2, y2, z2)
    glVertex3f(x1, y2, z2)
    glNormal3f(0.0, 0.0, -1.0)
    glVertex3f(x1, y2, z1)
    glVertex3f(x2, y2, z1)
    glVertex3f(x2, y1, z1)
    glVertex3f(x1, y1, z1)
    glNormal3f(0.0, -1.0, 0.0)
    glVertex3f(x1, y1, z1)
    glVertex3f(x2, y1, z1)
    glVertex3f(x2, y1, z2)
    glVertex3f(x1, y1, z2)
    glNormal3f(0.0, 1.0, 0.0)
    glVertex3f(x2, y2, z1)
    glVertex3f(x1, y2, z1)
    glVertex3f(x1, y2, z2)
    glVertex3f(x2, y2, z2)
    glNormal3f(1.0, 0.0, 0.0)
    glVertex3f(x2, y1, z1)
    glVertex3f(x2, y2, z1)
    glVertex3f(x2, y2, z2)
    glVertex3f(x2, y1, z2)
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
    """Draw the 12 wireframe edges of a box.

    Args:
        x1-z2: Box extents.  r, g, b: Edge color in [0, 1].
    """
    glDisable(GL_LIGHTING)
    glColor3f(r, g, b)
    glBegin(GL_LINES)
    for _z in (z1, z2):
        glVertex3f(x1, y1, _z)
        glVertex3f(x2, y1, _z)
        glVertex3f(x2, y1, _z)
        glVertex3f(x2, y2, _z)
        glVertex3f(x2, y2, _z)
        glVertex3f(x1, y2, _z)
        glVertex3f(x1, y2, _z)
        glVertex3f(x1, y1, _z)
    for _x, _y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)):
        glVertex3f(_x, _y, z1)
        glVertex3f(_x, _y, z2)
    glEnd()
    glEnable(GL_LIGHTING)


def _draw_arm_3d(
    robot: object,
    q1: float,
    q2: float,
    z: float,
    link1_color: tuple[float, float, float],
    link2_color: tuple[float, float, float],
    link_radius: float = 0.04,
) -> None:
    """Draw the 3-D RRP arm as two thick lines with joint spheres.

    Renders the two links of the arm at height *z* as thicker GL_LINES
    and marks the base, shoulder joint, and end-effector with small boxes.

    Args:
        robot: :class:`~arco.kinematics.RRPRobot` instance.
        q1: First revolute joint angle (radians).
        q2: Second revolute joint angle (radians).
        z: Prismatic joint height (meters).
        link1_color: RGB colors for the shoulder-to-elbow link.
        link2_color: RGB colors for the elbow-to-EE link.
        link_radius: Half-size of joint marker boxes.
    """
    origin, j2, ee = robot.link_segments(q1, q2, z)  # type: ignore[attr-defined]
    ox, oy, oz = float(origin[0]), float(origin[1]), float(origin[2])
    jx, jy, jz = float(j2[0]), float(j2[1]), float(j2[2])
    ex, ey, ez_val = float(ee[0]), float(ee[1]), float(ee[2])

    glDisable(GL_LIGHTING)

    # Prismatic (P) axis indicator from world origin to current arm base.
    glLineWidth(4.0)
    glColor3f(0.92, 0.92, 0.96)
    glBegin(GL_LINES)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(0.0, 0.0, oz)
    glEnd()

    glLineWidth(6.0)

    glColor3f(*link1_color)
    glBegin(GL_LINES)
    glVertex3f(ox, oy, oz)
    glVertex3f(jx, jy, jz)
    glEnd()

    glColor3f(*link2_color)
    glBegin(GL_LINES)
    glVertex3f(jx, jy, jz)
    glVertex3f(ex, ey, ez_val)
    glEnd()

    glLineWidth(1.0)
    glEnable(GL_LIGHTING)

    r = link_radius
    # Base (origin)
    _draw_box(ox - r, oy - r, oz - r, ox + r, oy + r, oz + r, 0.8, 0.8, 0.8)
    # Elbow
    _draw_box(jx - r, jy - r, jz - r, jx + r, jy + r, jz + r, *link1_color)
    # End-effector
    _draw_box(
        ex - r * 1.5,
        ey - r * 1.5,
        ez_val - r * 1.5,
        ex + r * 1.5,
        ey + r * 1.5,
        ez_val + r * 1.5,
        *link2_color,
    )


def _draw_workspace_cylinder(
    r_min: float, r_max: float, z_min: float, z_max: float, segments: int = 48
) -> None:
    """Draw the cylindrical workspace boundary as thin wireframe rings.

    Args:
        r_min: Inner annulus radius in metres.
        r_max: Outer annulus radius in metres.
        z_min: Bottom of workspace in metres.
        z_max: Top of workspace in metres.
        segments: Number of polygon segments per ring.
    """
    glDisable(GL_LIGHTING)
    glColor3f(*_C_GRID)
    glLineWidth(0.7)
    for z in (z_min, z_max):
        for r in (r_min, r_max):
            glBegin(GL_LINE_LOOP)
            for i in range(segments):
                ang = 2.0 * math.pi * i / segments
                glVertex3f(r * math.cos(ang), r * math.sin(ang), z)
            glEnd()
    glLineWidth(1.0)
    glEnable(GL_LIGHTING)


def _draw_grid_xy(r_max: float) -> None:
    """Draw a flat XY ground-plane reference grid up to radius *r_max*.

    Args:
        r_max: Draw grid lines within ±*r_max* in both X and Y.
    """
    spacing = 0.5
    glDisable(GL_LIGHTING)
    glColor3f(*_C_GRID)
    glBegin(GL_LINES)
    x = -r_max
    while x <= r_max + 1e-6:
        glVertex3f(x, -r_max, 0.0)
        glVertex3f(x, r_max, 0.0)
        x += spacing
    y = -r_max
    while y <= r_max + 1e-6:
        glVertex3f(-r_max, y, 0.0)
        glVertex3f(r_max, y, 0.0)
        y += spacing
    glEnd()
    glEnable(GL_LIGHTING)


def _draw_path_3d(
    path: list[np.ndarray], robot: object, r: float, g: float, b: float
) -> None:
    """Convert a 3-D joint-space path to Cartesian and draw it.

    Each waypoint ``[q1, q2, z]`` is mapped via FK to the end-effector
    position ``(x, y, z)`` which is then drawn as a GL_LINE_STRIP.

    Args:
        path: Ordered list of ``[q1, q2, z]`` arrays.
        robot: :class:`~arco.kinematics.RRPRobot` instance.
        r, g, b: Line colour in [0, 1].
    """
    if len(path) < 2:
        return
    glDisable(GL_LIGHTING)
    glLineWidth(3.0)
    glColor3f(r, g, b)
    glBegin(GL_LINE_STRIP)
    for pt in path:
        fk = robot.forward_kinematics(float(pt[0]), float(pt[1]), float(pt[2]))  # type: ignore[attr-defined]
        glVertex3f(float(fk[0]), float(fk[1]), float(fk[2]))
    glEnd()
    glLineWidth(1.0)
    glEnable(GL_LIGHTING)


def _draw_lookahead_3d(pos: np.ndarray, r: float, g: float, b: float) -> None:
    """Draw a small glowing lookahead marker at *pos*.

    Args:
        pos: 3-D Cartesian position (x, y, z).
        r, g, b: Marker colour in [0, 1].
    """
    h = 0.08
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    _draw_box(x - h, y - h, z - h, x + h, y + h, z + h, r, g, b)
    _draw_box_edges(x - h, y - h, z - h, x + h, y + h, z + h, r, g, b)


# ---------------------------------------------------------------------------
# HUD overlay helpers
# ---------------------------------------------------------------------------


def _format_clock(seconds: float) -> str:
    """Format seconds as ``MMminSSs`` rounded to whole seconds."""
    rounded = int(round(seconds))
    mins, secs = divmod(rounded, 60)
    return f"{mins:02d}min{secs:02d}s"


def _blit_overlay(
    src: pygame.Surface,
    px: int,
    py: int,
    sw: int,
    sh: int,
) -> None:
    """Upload *src* as an RGBA texture and draw it as a screen-aligned quad."""
    w, h = src.get_size()
    data = pygame.image.tostring(src, "RGBA", True)

    tex_id = int(glGenTextures(1))
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data
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


def _status_surface(
    font: pygame.font.Font,
    rrt_metrics: dict,
    sst_metrics: dict,
    phase: str,
    hold_remaining: float,
) -> pygame.Surface:
    """Build a semi-transparent status panel surface.

    Args:
        font: Monospace font.
        rrt_metrics: RRT* metrics dict.
        sst_metrics: SST metrics dict.
        phase: Current phase (``"show"``, ``"race"``, or ``"done"``).
        hold_remaining: Seconds until race starts.

    Returns:
        RGBA pygame surface.
    """

    def _planner_lines(name: str, metrics: dict) -> list[str]:
        return [
            name,
            (
                "  Planner steps / nodes: "
                f"{int(metrics['steps'])} / {int(metrics['nodes'])}"
            ),
            f"  Planner time: {_format_clock(float(metrics['planner_time']))}",
            (
                "  Planned path length: "
                f"{float(metrics['planned_path_length']):.2f} rad+m"
            ),
            (
                "  Trajectory arc length: "
                f"{float(metrics['trajectory_arc_length']):.2f} rad+m"
            ),
            (
                "  Trajectory duration: "
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
    """Build a one-line keyboard hint surface."""
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
    """Build a translucent centered winner-banner surface."""
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
    """Return cumulative arc lengths along *path* starting at 0.0."""
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

    Returns ``(config, at_goal)`` — interpolated ``[q1, q2, z]`` config
    and a flag that is ``True`` once the end of the path is reached.
    """
    if dist >= arcs[-1]:
        return path[-1].copy(), True
    for i in range(len(arcs) - 1):
        if arcs[i + 1] >= dist:
            seg = arcs[i + 1] - arcs[i]
            t = (dist - arcs[i]) / max(seg, _EPSILON)
            return path[i] + t * (path[i + 1] - path[i]), False
    return path[-1].copy(), True


# ---------------------------------------------------------------------------
# Race robot (joint-space P-controller)
# ---------------------------------------------------------------------------


class RRPRaceRobot:
    """Simple P-controller tracking the planning path in joint space.

    Each DOF is controlled with independent velocity and acceleration limits:
    revolute joints (q1, q2) use angular units (rad/s, rad/s²) while the
    prismatic joint (z) uses linear units (m/s, m/s²).

    Args:
        start: Initial joint config ``[q1, q2, z]``.
        max_vel_ang: Maximum angular velocity for q1/q2 joints (rad/s).
        max_vel_lin: Maximum linear velocity for z joint (m/s).
        max_acc_ang: Maximum angular acceleration for q1/q2 joints (rad/s²).
        max_acc_lin: Maximum linear acceleration for z joint (m/s²).
        proportional_gain: P-gain.
    """

    def __init__(
        self,
        start: np.ndarray,
        max_vel_ang: float = 1.5,
        max_vel_lin: float = 1.0,
        max_acc_ang: float = 3.0,
        max_acc_lin: float = 2.0,
        proportional_gain: float = 2.5,
    ) -> None:
        self.q: np.ndarray = start.astype(float).copy()
        self.vel: np.ndarray = np.zeros(3)
        self._max_vel = np.array([max_vel_ang, max_vel_ang, max_vel_lin])
        self._max_acc = np.array([max_acc_ang, max_acc_ang, max_acc_lin])
        self._k_p = proportional_gain

    def step(self, target: np.ndarray, dt: float) -> None:
        """Advance one timestep toward *target* with per-DOF limits.

        Each component is clipped independently: angular limits apply to
        q1/q2 and linear limits apply to z.

        Args:
            target: Carrot joint config ``[q1, q2, z]``.
            dt: Integration timestep in seconds.
        """
        err = target - self.q
        desired_vel = np.clip(self._k_p * err, -self._max_vel, self._max_vel)
        dv = desired_vel - self.vel
        dv = np.clip(dv, -self._max_acc * dt, self._max_acc * dt)
        self.vel = np.clip(self.vel + dv, -self._max_vel, self._max_vel)
        self.q = self.q + self.vel * dt


# ---------------------------------------------------------------------------
# Race simulation
# ---------------------------------------------------------------------------


def run_race(
    scene: RRPScene,
    cfg: dict,
    *,
    fps: int = 30,
    dt: float = 0.05,
    record: str = "",
    record_duration: float = 90.0,
) -> None:
    """Run the 3-D RRP arm race with OpenGL rendering.

    Phase 1 — **path reveal**: both paths shown for :data:`_HOLD_SECS` s.
    Phase 2 — **race**: both arms advance simultaneously.
    Phase 3 — **post-finish**: once both robots reach the goal, the
    simulation continues for :data:`_POST_FINISH_SECS` seconds.

    Args:
        scene: Fully built :class:`~scenes.rrp.RRPScene`.
        cfg: Configuration dict loaded from ``tools/config/rrp.yml``.
        fps: Target frame rate in frames per second.
        dt: Simulation timestep in seconds.
        record: Output MP4 path.  Empty string = interactive mode.
        record_duration: Maximum recording length in seconds.
    """
    recording = bool(record)
    max_frames = int(fps * record_duration)

    sim_cfg = cfg.get("simulator", cfg)
    race_speed = float(sim_cfg.get("race_speed", 0.6))
    max_ang_vel = float(sim_cfg.get("max_ang_vel", 1.5))
    max_lin_vel = float(sim_cfg.get("max_lin_vel", 1.0))
    max_ang_acc = float(sim_cfg.get("max_ang_acc", 3.0))
    max_lin_acc = float(sim_cfg.get("max_lin_acc", 2.0))
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

    run_with_loading_screen(scene, sw, sh, bg_color=(18, 22, 32))
    pygame.display.set_caption(scene.title)

    font = pygame.font.SysFont("monospace", 14)
    big_font = pygame.font.SysFont("monospace", 40, bold=True)
    hint_surf = _hint_surface(font)

    camera = Camera3D()
    robot = scene.robot
    rrt_path = scene.rrt_path
    sst_path = scene.sst_path
    rrt_nav = scene.rrt_traj if scene.rrt_traj else rrt_path
    sst_nav = scene.sst_traj if scene.sst_traj else sst_path
    obstacles = scene.obstacles
    rrt_metrics = scene.rrt_metrics
    sst_metrics = scene.sst_metrics

    rrt_arcs = _arc_lengths(rrt_nav) if rrt_nav else [0.0]
    sst_arcs = _arc_lengths(sst_nav) if sst_nav else [0.0]

    goal_reach_dist = float(
        cfg.get("planner", cfg).get("goal_tolerance", 0.30)
    )

    hold_timer = 0.0
    _max_vel_rrp = np.array([max_ang_vel, max_ang_vel, max_lin_vel])
    _max_acc_rrp = np.array([max_ang_acc, max_ang_acc, max_lin_acc])
    rrt_robot = JointSpaceTracker(
        max_vel=_max_vel_rrp,
        max_acc=_max_acc_rrp,
        occupancy=scene.occ,
        repulsion_gain=repulsion_gain,
    )
    rrt_robot.reset(scene.start_q)
    sst_robot = JointSpaceTracker(
        max_vel=_max_vel_rrp,
        max_acc=_max_acc_rrp,
        occupancy=scene.occ,
        repulsion_gain=repulsion_gain,
    )
    sst_robot.reset(scene.start_q)
    rrt_carrot_dist = 0.0
    sst_carrot_dist = 0.0
    rrt_carrot = scene.start_q.copy()
    sst_carrot = scene.start_q.copy()
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
            # --- Events ---
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
                        rrt_robot.reset(scene.start_q)
                        sst_robot.reset(scene.start_q)
                        rrt_carrot_dist = 0.0
                        sst_carrot_dist = 0.0
                        rrt_carrot = scene.start_q.copy()
                        sst_carrot = scene.start_q.copy()
                        rrt_trail = [scene.start.copy()]
                        sst_trail = [scene.start.copy()]
                        rrt_done = sst_done = False
                        winner = ""
                        post_timer = 0.0
                        phase = "show"
                        paused = False

            # --- Camera controls ---
            if not paused:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LEFT]:
                    camera.azim -= _CAM_ROT_SPEED * dt
                if keys[pygame.K_RIGHT]:
                    camera.azim += _CAM_ROT_SPEED * dt
                if keys[pygame.K_UP]:
                    camera.elev = min(
                        math.radians(80), camera.elev + _CAM_ROT_SPEED * dt
                    )
                if keys[pygame.K_DOWN]:
                    camera.elev = max(
                        math.radians(5), camera.elev - _CAM_ROT_SPEED * dt
                    )
                if keys[pygame.K_PLUS] or keys[pygame.K_EQUALS]:
                    camera.dist = max(
                        0.5, camera.dist * (1.0 - _CAM_ZOOM_STEP)
                    )
                if keys[pygame.K_MINUS]:
                    camera.dist *= 1.0 + _CAM_ZOOM_STEP
                if recording:
                    camera.azim += _CAM_AUTO_ROT * dt

            # --- Simulation step ---
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
                        if rrt_lag < 0.3:
                            rrt_carrot_dist = min(
                                rrt_carrot_dist + race_speed * dt,
                                rrt_arcs[-1],
                            )
                        rrt_carrot, _ = _path_pos(
                            rrt_nav, rrt_arcs, rrt_carrot_dist
                        )
                        rrt_robot.step(rrt_carrot, dt)
                        fk = robot.forward_kinematics(
                            float(rrt_robot.q[0]),
                            float(rrt_robot.q[1]),
                            float(rrt_robot.q[2]),
                        )
                        rrt_trail.append(np.array([fk[0], fk[1], fk[2]]))
                        rrt_done = (
                            float(np.linalg.norm(rrt_robot.q - scene.goal_q))
                            < goal_reach_dist
                        )
                    if sst_nav and not sst_done:
                        sst_lag = float(
                            np.linalg.norm(sst_robot.q - sst_carrot)
                        )
                        if sst_lag < 0.3:
                            sst_carrot_dist = min(
                                sst_carrot_dist + race_speed * dt,
                                sst_arcs[-1],
                            )
                        sst_carrot, _ = _path_pos(
                            sst_nav, sst_arcs, sst_carrot_dist
                        )
                        sst_robot.step(sst_carrot, dt)
                        fk = robot.forward_kinematics(
                            float(sst_robot.q[0]),
                            float(sst_robot.q[1]),
                            float(sst_robot.q[2]),
                        )
                        sst_trail.append(np.array([fk[0], fk[1], fk[2]]))
                        sst_done = (
                            float(np.linalg.norm(sst_robot.q - scene.goal_q))
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

            # --- 3-D render (OpenGL) ---
            from OpenGL.GL import GL_DEPTH_BUFFER_BIT

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            _set_camera(camera)

            _draw_grid_xy(robot.workspace_radius())
            _draw_workspace_cylinder(
                robot.workspace_annulus()[0],
                robot.workspace_radius(),
                robot.z_min,
                robot.z_max,
            )

            # Obstacles
            for obs in obstacles:
                if _is_pillar(obs, z_max_full=robot.z_max):
                    fc, ec = _C_PILLAR, _C_PILLAR_EDGE
                else:
                    fc, ec = _C_SLAB, _C_SLAB_EDGE
                _draw_box(*obs, *fc)
                _draw_box_edges(*obs, *ec)

            # Planned paths (joint-space → Cartesian via FK)
            if rrt_path:
                _draw_path_3d(rrt_path, robot, *_C_RRT)
            if scene.rrt_traj:
                _draw_path_3d(scene.rrt_traj, robot, *_C_TRAJ_RRT)
            if sst_path:
                _draw_path_3d(sst_path, robot, *_C_SST)
            if scene.sst_traj:
                _draw_path_3d(scene.sst_traj, robot, *_C_TRAJ_SST)

            # Trail traces (actual Cartesian EE positions)
            if phase in ("race", "done"):
                for trail, col in (
                    (rrt_trail, _C_TRAIL_RRT),
                    (sst_trail, _C_TRAIL_SST),
                ):
                    if len(trail) >= 2:
                        glDisable(GL_LIGHTING)
                        glLineWidth(5.0)
                        glColor3f(*col)
                        glBegin(GL_LINE_STRIP)
                        for _pt in trail:
                            glVertex3f(
                                float(_pt[0]), float(_pt[1]), float(_pt[2])
                            )
                        glEnd()
                        glLineWidth(1.0)
                        glEnable(GL_LIGHTING)

            # Start / goal markers
            for pad_q, pad_col in (
                (scene.start_q, _C_START),
                (scene.goal_q, _C_GOAL),
            ):
                fk = robot.forward_kinematics(
                    float(pad_q[0]), float(pad_q[1]), float(pad_q[2])
                )
                px, py, pz = float(fk[0]), float(fk[1]), float(fk[2])
                _draw_box(
                    px - 0.08,
                    py - 0.08,
                    pz - 0.02,
                    px + 0.08,
                    py + 0.08,
                    pz + 0.02,
                    *pad_col,
                )

            # Robot arms
            if phase in ("race", "done"):
                if phase == "race":
                    if rrt_path and not rrt_done:
                        fk_cr = robot.forward_kinematics(
                            float(rrt_carrot[0]),
                            float(rrt_carrot[1]),
                            float(rrt_carrot[2]),
                        )
                        _draw_lookahead_3d(
                            np.array([fk_cr[0], fk_cr[1], fk_cr[2]]),
                            *_C_LA_RRT,
                        )
                    if sst_path and not sst_done:
                        fk_cs = robot.forward_kinematics(
                            float(sst_carrot[0]),
                            float(sst_carrot[1]),
                            float(sst_carrot[2]),
                        )
                        _draw_lookahead_3d(
                            np.array([fk_cs[0], fk_cs[1], fk_cs[2]]),
                            *_C_LA_SST,
                        )
                _draw_arm_3d(
                    robot,
                    float(rrt_robot.q[0]),
                    float(rrt_robot.q[1]),
                    float(rrt_robot.q[2]),
                    _C_LINK1_RRT,
                    _C_LINK2_RRT,
                )
                _draw_arm_3d(
                    robot,
                    float(sst_robot.q[0]),
                    float(sst_robot.q[1]),
                    float(sst_robot.q[2]),
                    _C_LINK1_SST,
                    _C_LINK2_SST,
                )

            # --- 2-D HUD overlay ---
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


def main(cfg: dict) -> None:
    """Parse CLI arguments and launch the RRP 3-D OpenGL race simulation."""
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

    scene = RRPScene(cfg)
    run_race(
        scene,
        cfg,
        fps=args.fps,
        record=args.record,
        record_duration=args.record_duration,
    )


if __name__ == "__main__":
    import argparse as _argparse

    import yaml as _yaml

    _parser = _argparse.ArgumentParser()
    _parser.add_argument("scenario", metavar="FILE")
    _args, _rest = _parser.parse_known_args()
    with open(_args.scenario) as _fh:
        _cfg = _yaml.safe_load(_fh) or {}
    import sys as _sys

    _sys.argv = [_sys.argv[0], *_rest]
    main(_cfg)
