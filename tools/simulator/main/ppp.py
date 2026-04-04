#!/usr/bin/env python
"""RRT* vs SST race in a 3-D warehouse PPP robot environment.

Both planners compete on the same 3-D obstacle map.  A full-width blocking
wall forces both planners to arc over the top (z > 4.5 m clearance) rather
than squeezing around the sides.  Once both paths are found the end-effectors
race from start to goal — the first to arrive wins.

The exploration tree is **not** rendered; only the chosen paths are shown,
keeping the 3-D view uncluttered.

Camera controls
---------------
LEFT / RIGHT   Rotate camera azimuth
UP / DOWN      Change camera elevation
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

Optional flags::

    python main/ppp.py --fps 30
    python main/ppp.py --record /tmp/ppp.mp4 --record-duration 90
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

from config import load_config
from scenes.ppp import PPPScene
from sim.video import VideoWriter

logger = logging.getLogger(__name__)

_DEFAULT_SCREEN_W = 1280
_DEFAULT_SCREEN_H = 800

# Simulation parameters
_HOLD_SECS: float = 2.0  # show paths before race starts
_POST_FINISH_SECS: float = 2.5  # linger after last effector reaches goal
_RACE_SPEED: float = 3.0  # m/s along path

# Camera orbit speeds (radians per second)
_CAM_ROT_SPEED: float = math.radians(40)
_CAM_AUTO_ROT: float = math.radians(6)  # slow orbit during recording
_CAM_ZOOM_SPEED: float = 0.05  # fraction of dist per second

# Default camera parameters
_CAM_AZIM: float = math.radians(38)
_CAM_ELEV: float = math.radians(28)
_CAM_DIST: float = 36.0
_CAM_FOV: float = 580.0

# World centre of the workspace (10 m × 5 m × 1.5 m for a 20×10×6 box)
_WS_CENTER: np.ndarray = np.array([10.0, 5.0, 1.5])

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_C_BG: tuple[int, int, int] = (18, 22, 32)
_C_GRID: tuple[int, int, int] = (38, 44, 56)
_C_WALL: tuple[int, int, int] = (170, 80, 45)
_C_BOX: tuple[int, int, int] = (145, 105, 60)
_C_WALL_EDGE: tuple[int, int, int] = (90, 42, 22)
_C_BOX_EDGE: tuple[int, int, int] = (80, 58, 32)
_C_RRT_PATH: tuple[int, int, int] = (100, 165, 255)
_C_SST_PATH: tuple[int, int, int] = (60, 230, 190)
_C_RRT_EFF: tuple[int, int, int] = (80, 145, 255)
_C_SST_EFF: tuple[int, int, int] = (40, 210, 170)
_C_START: tuple[int, int, int] = (55, 220, 85)
_C_GOAL: tuple[int, int, int] = (220, 75, 220)
_C_HUD: tuple[int, int, int] = (220, 220, 220)
_C_HUD_DIM: tuple[int, int, int] = (130, 130, 130)
_C_HUD_SHADOW: tuple[int, int, int] = (30, 35, 45)
_C_WINNER: tuple[int, int, int] = (255, 215, 50)
_C_TIE: tuple[int, int, int] = (200, 200, 80)

# End-effector half-dimensions (metres)
_EFF_HXY: float = 0.25  # half width / depth (square face)
_EFF_HZ: float = 0.40  # half height

# Box face brightness factors (index matches _BOX_FACES order)
# Order: bottom, top, front(-y), back(+y), left(-x), right(+x)
_FACE_BRIGHT: list[float] = [0.45, 1.0, 0.70, 0.60, 0.80, 0.75]


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


class Camera3D:
    """Perspective orbit camera for 3-D pygame rendering.

    The camera orbits around a fixed world centre point.  Azimuth rotates
    around the world Z-axis (up); elevation tilts the view up or down.

    Args:
        azim: Initial azimuth angle in radians.
        elev: Initial elevation angle in radians (0 = horizontal).
        dist: Initial distance from :attr:`center` in metres.
        fov: Focal length in pixels.
        center: World point the camera orbits around.
    """

    def __init__(
        self,
        azim: float = _CAM_AZIM,
        elev: float = _CAM_ELEV,
        dist: float = _CAM_DIST,
        fov: float = _CAM_FOV,
        center: np.ndarray | None = None,
    ) -> None:
        self.azim = azim
        self.elev = min(math.radians(80), max(math.radians(5), elev))
        self.dist = dist
        self.fov = fov
        self.center = (
            center.copy() if center is not None else _WS_CENTER.copy()
        )

    def project(
        self,
        pts: np.ndarray,
        screen_w: int,
        screen_h: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project world points to screen pixels using perspective division.

        The camera sits on a sphere of radius :attr:`dist` centred on
        :attr:`center`.  Azimuth rotates around the world Z-axis and
        elevation tilts up from the horizontal plane.

        Args:
            pts: World positions of shape ``(N, 3)`` or ``(3,)``.
            screen_w: Screen width in pixels.
            screen_h: Screen height in pixels.

        Returns:
            Tuple ``(screen_xy, cam_z)`` where *screen_xy* is ``(N, 2)``
            integer-like pixel coordinates and *cam_z* is the signed depth
            in camera space (larger = further from camera).
        """
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        rel = pts - self.center

        ca = math.cos(self.azim)
        sa = math.sin(self.azim)
        ce = math.cos(self.elev)
        se = math.sin(self.elev)

        # Rotate around world Z (azimuth)
        x1 = rel[:, 0] * ca - rel[:, 1] * sa
        y1 = rel[:, 0] * sa + rel[:, 1] * ca
        z1 = rel[:, 2]

        # Rotate around camera X (elevation — brings Z into view)
        # After azimuth, camera looks along +y1; elevation tilts up.
        x2 = x1
        y2 = y1 * ce - z1 * se
        z2 = y1 * se + z1 * ce

        # Camera is behind the scene along the +y2 axis.
        depth = np.where(self.dist - y2 < 0.1, 0.1, self.dist - y2)

        sx = screen_w / 2.0 + self.fov * x2 / depth
        sy = screen_h / 2.0 - self.fov * z2 / depth

        return np.column_stack([sx, sy]), y2


# ---------------------------------------------------------------------------
# Box face geometry
# ---------------------------------------------------------------------------

# Eight corners of a canonical unit cube (indices 0-7).
# Ordering: 0-3 = bottom face (z=z1), 4-7 = top face (z=z2).
_BOX_FACES: list[list[int]] = [
    [0, 1, 2, 3],  # bottom  (−z)
    [4, 5, 6, 7],  # top     (+z)
    [0, 1, 5, 4],  # front   (−y)
    [2, 3, 7, 6],  # back    (+y)
    [0, 3, 7, 4],  # left    (−x)
    [1, 2, 6, 5],  # right   (+x)
]


def _box_corners(box: tuple[float, ...]) -> np.ndarray:
    """Return the 8 corners of a box as shape ``(8, 3)``.

    Args:
        box: ``(x1, y1, z1, x2, y2, z2)`` extents.

    Returns:
        Array of 8 corner positions, shape ``(8, 3)``.
    """
    x1, y1, z1, x2, y2, z2 = box
    return np.array(
        [
            [x1, y1, z1],
            [x2, y1, z1],
            [x2, y2, z1],
            [x1, y2, z1],
            [x1, y1, z2],
            [x2, y1, z2],
            [x2, y2, z2],
            [x1, y2, z2],
        ],
        dtype=float,
    )


def _tint(
    color: tuple[int, int, int], factor: float
) -> tuple[int, int, int]:
    """Scale a colour by *factor*, clamping to [0, 255].

    Args:
        color: Source RGB colour.
        factor: Brightness multiplier.

    Returns:
        Adjusted RGB colour.
    """
    return (
        max(0, min(255, int(color[0] * factor))),
        max(0, min(255, int(color[1] * factor))),
        max(0, min(255, int(color[2] * factor))),
    )


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def draw_box(
    surface: pygame.Surface,
    box: tuple[float, ...],
    camera: Camera3D,
    color: tuple[int, int, int],
    edge_color: tuple[int, int, int] | None = None,
) -> None:
    """Draw a filled, depth-sorted box (painter's algorithm).

    Args:
        surface: Pygame surface to draw onto.
        box: ``(x1, y1, z1, x2, y2, z2)`` extents.
        camera: Active 3-D camera.
        color: Base fill colour; each face is tinted by
            :data:`_FACE_BRIGHT`.
        edge_color: Outline colour.  ``None`` skips edge drawing.
    """
    sw, sh = surface.get_size()
    corners = _box_corners(box)
    screen_xy, _ = camera.project(corners, sw, sh)
    screen_pts = screen_xy.astype(int)

    # Sort faces back-to-front using the depth of each face centroid.
    centroids = np.array([corners[f].mean(axis=0) for f in _BOX_FACES])
    _, depths = camera.project(centroids, sw, sh)
    order = np.argsort(depths)[::-1]

    for fi in order:
        face = _BOX_FACES[fi]
        poly = [screen_pts[j].tolist() for j in face]
        face_color = _tint(color, _FACE_BRIGHT[fi])
        pygame.draw.polygon(surface, face_color, poly)
        if edge_color is not None:
            pygame.draw.polygon(surface, edge_color, poly, 1)


def draw_path_3d(
    surface: pygame.Surface,
    path: list[np.ndarray],
    camera: Camera3D,
    color: tuple[int, int, int],
    line_width: int = 2,
) -> None:
    """Draw a 3-D path as connected line segments.

    Args:
        surface: Pygame surface.
        path: Ordered list of 3-D waypoints.
        camera: Active 3-D camera.
        color: Line colour.
        line_width: Line width in pixels.
    """
    if len(path) < 2:
        return
    sw, sh = surface.get_size()
    screen_xy, _ = camera.project(np.array(path), sw, sh)
    pts = screen_xy.astype(int)
    for i in range(len(pts) - 1):
        pygame.draw.line(surface, color, pts[i], pts[i + 1], line_width)


def draw_effector(
    surface: pygame.Surface,
    pos: np.ndarray,
    camera: Camera3D,
    color: tuple[int, int, int],
) -> None:
    """Draw the PPP end-effector as a small coloured parallelepiped.

    Args:
        surface: Pygame surface.
        pos: End-effector centre (x, y, z).
        camera: Active 3-D camera.
        color: Fill colour.
    """
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    box = (
        x - _EFF_HXY,
        y - _EFF_HXY,
        z,
        x + _EFF_HXY,
        y + _EFF_HXY,
        z + 2.0 * _EFF_HZ,
    )
    draw_box(surface, box, camera, color, edge_color=(230, 230, 230))


def draw_floor_grid(
    surface: pygame.Surface,
    camera: Camera3D,
    x_max: float,
    y_max: float,
    spacing: float = 2.0,
) -> None:
    """Draw a ground-plane reference grid for depth cues.

    Args:
        surface: Pygame surface.
        camera: Active 3-D camera.
        x_max: Grid extent along x (metres).
        y_max: Grid extent along y (metres).
        spacing: Distance between grid lines (metres).
    """
    sw, sh = surface.get_size()
    xs = np.arange(0.0, x_max + spacing, spacing)
    ys = np.arange(0.0, y_max + spacing, spacing)

    for x in xs:
        sc, _ = camera.project(
            np.array([[x, 0.0, 0.0], [x, y_max, 0.0]]), sw, sh
        )
        pygame.draw.line(
            surface, _C_GRID, sc[0].astype(int), sc[1].astype(int), 1
        )
    for y in ys:
        sc, _ = camera.project(
            np.array([[0.0, y, 0.0], [x_max, y, 0.0]]), sw, sh
        )
        pygame.draw.line(
            surface, _C_GRID, sc[0].astype(int), sc[1].astype(int), 1
        )


def _blit_text(
    surface: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    color: tuple[int, int, int],
    x: int,
    y: int,
) -> int:
    """Render a shadowed text line and return the next y position.

    Args:
        surface: Pygame surface.
        font: Pygame font.
        text: Text to render.
        color: Text colour.
        x: Left x pixel position.
        y: Top y pixel position.

    Returns:
        Y position of the line below the rendered text.
    """
    shadow = font.render(text, True, _C_HUD_SHADOW)
    surface.blit(shadow, (x + 1, y + 1))
    rendered = font.render(text, True, color)
    surface.blit(rendered, (x, y))
    return y + font.get_linesize() + 2


def _blit_center(
    surface: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    color: tuple[int, int, int],
    y: int,
) -> None:
    """Render a centred shadowed line at vertical position *y*.

    Args:
        surface: Pygame surface.
        font: Pygame font.
        text: Text to render.
        color: Text colour.
        y: Vertical pixel position.
    """
    rendered = font.render(text, True, color)
    x = (surface.get_width() - rendered.get_width()) // 2
    shadow = font.render(text, True, _C_HUD_SHADOW)
    surface.blit(shadow, (x + 1, y + 1))
    surface.blit(rendered, (x, y))


def _draw_winner_banner(
    surface: pygame.Surface,
    big_font: pygame.font.Font,
    text: str,
    color: tuple[int, int, int],
) -> None:
    """Draw a translucent centred banner with large winner text.

    Args:
        surface: Pygame surface.
        big_font: Large pygame font.
        text: Banner text (e.g. ``"RRT* WINS!"``).
        color: Text colour.
    """
    rendered = big_font.render(text, True, color)
    rw, rh = rendered.get_width(), rendered.get_height()
    pad = 16
    bx = (surface.get_width() - rw) // 2 - pad
    by = (surface.get_height() - rh) // 2 - pad
    banner = pygame.Surface((rw + 2 * pad, rh + 2 * pad), pygame.SRCALPHA)
    banner.fill((10, 12, 22, 210))
    surface.blit(banner, (bx, by))
    surface.blit(rendered, (bx + pad, by + pad))


# ---------------------------------------------------------------------------
# Path interpolation
# ---------------------------------------------------------------------------


def _arc_lengths(path: list[np.ndarray]) -> list[float]:
    """Compute cumulative arc lengths along a path.

    Args:
        path: Ordered list of 3-D waypoints.

    Returns:
        List of cumulative distances: ``[0, d01, d01+d12, …, total]``.
    """
    lengths = [0.0]
    for i in range(len(path) - 1):
        lengths.append(
            lengths[-1]
            + float(np.linalg.norm(path[i + 1] - path[i]))
        )
    return lengths


def _path_position(
    path: list[np.ndarray],
    arc_lengths: list[float],
    distance: float,
) -> tuple[np.ndarray, bool]:
    """Interpolate a position along a path at arc-length *distance*.

    Args:
        path: Ordered list of waypoints.
        arc_lengths: Precomputed cumulative arc lengths (see
            :func:`_arc_lengths`).
        distance: Distance from the start of the path.

    Returns:
        ``(position, at_goal)`` — interpolated 3-D position and a flag
        set to ``True`` once the end of the path is reached.
    """
    total = arc_lengths[-1]
    if distance >= total:
        return path[-1].copy(), True
    for i in range(len(arc_lengths) - 1):
        if arc_lengths[i + 1] >= distance:
            seg = arc_lengths[i + 1] - arc_lengths[i]
            t = (distance - arc_lengths[i]) / max(seg, 1e-9)
            return path[i] + t * (path[i + 1] - path[i]), False
    return path[-1].copy(), True


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
    """Run the 3-D PPP warehouse race between RRT* and SST.

    Phase 1 — **path reveal**: both planned paths are displayed for
    :data:`_HOLD_SECS` seconds before the race begins.

    Phase 2 — **race**: both end-effectors advance along their paths
    at :data:`_RACE_SPEED` m/s simultaneously.  The simulation
    continues for :data:`_POST_FINISH_SECS` after the last effector
    reaches the goal, then exits.

    Args:
        scene: Fully built :class:`~scenes.ppp.PPPScene`.
        fps: Target frame rate (frames per second).
        dt: Simulation timestep in seconds.
        record: Output MP4 file path.  Empty string = interactive mode.
        record_duration: Maximum headless recording length (seconds).
    """
    recording = bool(record)
    max_record_frames = int(fps * record_duration)

    if recording:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    pygame.init()

    if recording:
        sw, sh = _DEFAULT_SCREEN_W, _DEFAULT_SCREEN_H
    else:
        info = pygame.display.Info()
        w = int(getattr(info, "current_w", 0) or 0)
        h = int(getattr(info, "current_h", 0) or 0)
        if w > 0 and h > 0:
            sw, sh = max(640, int(w * 0.9)), max(480, int(h * 0.9))
        else:
            sw, sh = _DEFAULT_SCREEN_W, _DEFAULT_SCREEN_H

    screen = pygame.display.set_mode((sw, sh))
    clock = pygame.time.Clock()

    # Build the scene (runs planners) — may take several seconds.
    scene.build()
    pygame.display.set_caption(scene.title)

    font = pygame.font.SysFont("monospace", 14)
    big_font = pygame.font.SysFont("monospace", 40, bold=True)

    camera = Camera3D()

    rrt_path = scene.rrt_path
    sst_path = scene.sst_path
    boxes = scene.boxes

    rrt_arcs = _arc_lengths(rrt_path) if rrt_path else [0.0]
    sst_arcs = _arc_lengths(sst_path) if sst_path else [0.0]
    rrt_total = rrt_arcs[-1]
    sst_total = sst_arcs[-1]

    # Detect which box is the "main wall" for different colouring.
    def _is_wall(box: tuple[float, ...]) -> bool:
        return (box[3] - box[0]) <= 2.5 and (box[4] - box[1]) >= 8.0

    # Race state
    hold_timer = 0.0
    rrt_dist = 0.0
    sst_dist = 0.0
    rrt_pos = scene.start.copy()
    sst_pos = scene.start.copy()
    rrt_done = False
    sst_done = False
    winner: str = ""
    post_timer = 0.0
    phase: str = "show"  # "show" → "race" → "done"
    paused = False
    frame_count = 0

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
                        hold_timer = 0.0
                        rrt_dist = 0.0
                        sst_dist = 0.0
                        rrt_pos = scene.start.copy()
                        sst_pos = scene.start.copy()
                        rrt_done = False
                        sst_done = False
                        winner = ""
                        post_timer = 0.0
                        phase = "show"
                        paused = False
                        camera = Camera3D()

            # --- Camera rotation (keys held) ----------------------------
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
                    camera.dist *= 1.0 - _CAM_ZOOM_SPEED
                if keys[pygame.K_MINUS]:
                    camera.dist *= 1.0 + _CAM_ZOOM_SPEED

                # Slow auto-orbit during recording.
                if recording:
                    camera.azim += _CAM_AUTO_ROT * dt

            # --- Simulation step ----------------------------------------
            if not paused:
                if phase == "show":
                    hold_timer += dt
                    if hold_timer >= _HOLD_SECS:
                        phase = "race"

                elif phase == "race":
                    if rrt_path is not None and not rrt_done:
                        rrt_dist += _RACE_SPEED * dt
                        rrt_pos, rrt_done = _path_position(
                            rrt_path, rrt_arcs, rrt_dist
                        )
                    if sst_path is not None and not sst_done:
                        sst_dist += _RACE_SPEED * dt
                        sst_pos, sst_done = _path_position(
                            sst_path, sst_arcs, sst_dist
                        )

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

            # --- Draw ---------------------------------------------------
            screen.fill(_C_BG)

            # Ground grid
            draw_floor_grid(screen, camera, x_max=20.0, y_max=10.0)

            # Obstacle boxes — depth-sorted (furthest first).
            box_centers = np.array(
                [
                    [
                        (b[0] + b[3]) / 2,
                        (b[1] + b[4]) / 2,
                        (b[2] + b[5]) / 2,
                    ]
                    for b in boxes
                ]
            )
            _, box_depths = camera.project(box_centers, sw, sh)
            for bi in np.argsort(box_depths)[::-1]:
                box = boxes[bi]
                if _is_wall(box):
                    draw_box(
                        screen, box, camera, _C_WALL, edge_color=_C_WALL_EDGE
                    )
                else:
                    draw_box(
                        screen, box, camera, _C_BOX, edge_color=_C_BOX_EDGE
                    )

            # Planned paths
            if rrt_path:
                draw_path_3d(screen, rrt_path, camera, _C_RRT_PATH, 2)
            if sst_path:
                draw_path_3d(screen, sst_path, camera, _C_SST_PATH, 2)

            # Start / goal pads
            start_pad = (
                scene.start[0] - 0.35,
                scene.start[1] - 0.35,
                0.0,
                scene.start[0] + 0.35,
                scene.start[1] + 0.35,
                0.08,
            )
            goal_pad = (
                scene.goal[0] - 0.35,
                scene.goal[1] - 0.35,
                0.0,
                scene.goal[0] + 0.35,
                scene.goal[1] + 0.35,
                0.08,
            )
            draw_box(screen, start_pad, camera, _C_START)
            draw_box(screen, goal_pad, camera, _C_GOAL)

            # End-effectors (shown during race and after)
            if phase in ("race", "done"):
                draw_effector(screen, rrt_pos, camera, _C_RRT_EFF)
                draw_effector(screen, sst_pos, camera, _C_SST_EFF)

            # --- HUD ----------------------------------------------------
            y = 10
            y = _blit_text(
                screen, font, "RRT*", _C_RRT_PATH, 10, y
            )
            rrt_pct = (
                min(100, int(100 * rrt_dist / max(rrt_total, 1e-6)))
                if rrt_path
                else 0
            )
            y = _blit_text(
                screen, font, f"  {rrt_pct:3d}% complete", _C_RRT_PATH, 10, y
            )
            y += 4
            y = _blit_text(
                screen, font, "SST", _C_SST_PATH, 10, y
            )
            sst_pct = (
                min(100, int(100 * sst_dist / max(sst_total, 1e-6)))
                if sst_path
                else 0
            )
            _blit_text(
                screen, font, f"  {sst_pct:3d}% complete", _C_SST_PATH, 10, y
            )

            # Controls hint
            hint = "← → ↑ ↓  rotate  |  +/-  zoom  |  SPACE  pause  |  R  restart"
            _blit_text(
                screen,
                font,
                hint,
                _C_HUD_DIM,
                sw // 2 - font.size(hint)[0] // 2,
                sh - font.get_linesize() - 6,
            )

            # Phase message
            if phase == "show":
                msg = (
                    f"Paths revealed — race starts in "
                    f"{max(0.0, _HOLD_SECS - hold_timer):.1f} s"
                )
                _blit_center(screen, font, msg, _C_HUD, sh - 30)

            # Winner banner
            if winner:
                w_color = _C_TIE if winner == "TIE!" else _C_WINNER
                _draw_winner_banner(screen, big_font, winner, w_color)

            pygame.display.flip()

            if video_writer is not None:
                video_writer.write_frame(screen)
                frame_count += 1
                if frame_count >= max_record_frames:
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
    """Parse arguments and start the PPP warehouse race simulation."""
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
        help="Record to this MP4 file instead of displaying interactively.",
    )
    parser.add_argument(
        "--record-duration",
        type=float,
        default=90.0,
        metavar="SECS",
        help="Maximum recording duration in seconds (default: 90).",
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
