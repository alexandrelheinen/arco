#!/usr/bin/env python
"""2-D RR robot arm simulator — split-screen Cartesian + joint space.

Shows a two-link planar arm tracking an optimized trajectory.  The
screen is divided into two panels:

* **Left 2/3** — Cartesian workspace: animated robot arm, end-effector
  trail, workspace annulus, rectangular obstacle, start and goal markers.
* **Right 1/3** — Joint C-space: collision configurations (gray scatter),
  trajectory trace in (theta1, theta2) space, live current-config dot.

The simulation first shows RRT* tracking for its full duration, then
switches to SST.  Both panels update in real time.

Keyboard controls
-----------------
SPACE         Pause / resume
R             Restart from the beginning
Q / Escape    Quit

Usage
-----
::

    cd tools/simulator
    python main/rr.py

Record a video::

    xvfb-run -a python main/rr.py --record /tmp/rr.mp4 --record-duration 90
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
import renderer_gl
from logging_config import configure_logging
from OpenGL.GL import (  # type: ignore[import-untyped]
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_LIGHTING,
    GL_LINE_LOOP,
    GL_LINE_STRIP,
    GL_LINES,
    GL_MODELVIEW,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_POINTS,
    GL_PROJECTION,
    GL_QUADS,
    GL_SCISSOR_TEST,
    GL_SRC_ALPHA,
    GL_TRIANGLE_FAN,
    glBegin,
    glBlendFunc,
    glClear,
    glClearColor,
    glColor4f,
    glDisable,
    glEnable,
    glEnd,
    glLineWidth,
    glLoadIdentity,
    glMatrixMode,
    glOrtho,
    glPointSize,
    glScissor,
    glVertex2f,
    glViewport,
)
from scenes.rr import RRScene
from sim.loading import run_with_loading_screen
from sim.video import VideoWriter

from config import load_config

logger = logging.getLogger(__name__)

_SW = 1280
_SH = 720

# Fraction of screen width for the Cartesian (left) panel.
_LEFT_FRAC = 2.0 / 3.0

# Post-finish hold time in seconds before switching planners.
_POST_FINISH_SECS = 1.5

# Minimum annulus inner radius below which the inner-hole punch-out is skipped.
_INNER_RADIUS_THRESHOLD: float = 1e-6

# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

_COLORS = load_config("colors")


def _rgb(section: str, key: str) -> tuple[int, int, int]:
    v = _COLORS[section][key]
    return (int(v[0]), int(v[1]), int(v[2]))


def _cf(t: tuple[int, int, int]) -> tuple[float, float, float]:
    return (t[0] / 255.0, t[1] / 255.0, t[2] / 255.0)


_C_BG = _rgb("map", "background")
_C_RRT_PATH = _rgb("rrt", "path")
_C_RRT_TRAJ = _rgb("rrt", "trajectory")
_C_SST_PATH = _rgb("sst", "path")
_C_SST_TRAJ = _rgb("sst", "trajectory")

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _cumulative_lengths(
    path: list[np.ndarray],
) -> list[float]:
    """Return cumulative arc lengths along *path*.

    Args:
        path: List of 2-element joint-config arrays.

    Returns:
        List of length ``len(path)`` starting at 0.0.
    """
    cum = [0.0]
    for i in range(1, len(path)):
        cum.append(cum[-1] + float(np.linalg.norm(path[i] - path[i - 1])))
    return cum


def _interp_path(
    path: list[np.ndarray], cum: list[float], s: float
) -> np.ndarray:
    """Interpolate along *path* at arc-length parameter *s*.

    Args:
        path: Ordered list of joint-config arrays.
        cum: Cumulative arc lengths (same length as *path*).
        s: Arc-length parameter in ``[0, cum[-1]]``.

    Returns:
        Interpolated joint config array of shape ``(2,)``.
    """
    s = max(0.0, min(s, cum[-1]))
    for i in range(1, len(cum)):
        if cum[i] >= s:
            seg = cum[i] - cum[i - 1]
            if seg < 1e-12:
                return path[i].copy()
            t = (s - cum[i - 1]) / seg
            return (1.0 - t) * path[i - 1] + t * path[i]
    return path[-1].copy()


# ---------------------------------------------------------------------------
# Viewport helpers
# ---------------------------------------------------------------------------

#: World-space half-size of the left Cartesian viewport in metres.
_CART_VIEWPORT_RADIUS: float = 2.0


def _set_left_viewport(sw: int, sh: int, left_w: int) -> None:
    """Configure GL viewport/projection for the left Cartesian panel.

    Args:
        sw: Total screen width in pixels.
        sh: Screen height in pixels.
        left_w: Width of the left panel in pixels.
    """
    glEnable(GL_SCISSOR_TEST)
    glScissor(0, 0, left_w, sh)
    glViewport(0, 0, left_w, sh)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    R = _CART_VIEWPORT_RADIUS
    asp = left_w / sh
    glOrtho(-R * asp, R * asp, -R, R, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def _set_right_viewport(sw: int, sh: int, left_w: int) -> None:
    """Configure GL viewport/projection for the right joint-space panel.

    Args:
        sw: Total screen width in pixels.
        sh: Screen height in pixels.
        left_w: Width of the left panel in pixels.
    """
    right_w = sw - left_w
    glEnable(GL_SCISSOR_TEST)
    glScissor(left_w, 0, right_w, sh)
    glViewport(left_w, 0, right_w, sh)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    Q = math.pi + 0.35
    glOrtho(-Q, Q, -Q, Q, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def _reset_viewport(sw: int, sh: int) -> None:
    """Reset GL viewport to the full screen.

    Args:
        sw: Screen width in pixels.
        sh: Screen height in pixels.
    """
    glDisable(GL_SCISSOR_TEST)
    glViewport(0, 0, sw, sh)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, sw, 0, sh, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


# ---------------------------------------------------------------------------
# Cartesian-panel drawing helpers
# ---------------------------------------------------------------------------


def _draw_workspace_annulus(
    r_min: float, r_max: float, segs: int = 64
) -> None:
    """Draw the reachable workspace as a filled annulus.

    Args:
        r_min: Inner radius in metres.
        r_max: Outer radius in metres.
        segs: Number of polygon segments.
    """
    glColor4f(0.27, 0.51, 0.71, 0.15)
    glBegin(GL_TRIANGLE_FAN)
    glVertex2f(0.0, 0.0)
    for i in range(segs + 1):
        a = 2.0 * math.pi * i / segs
        glVertex2f(r_max * math.cos(a), r_max * math.sin(a))
    glEnd()
    # Punch out the inner hole
    if r_min > _INNER_RADIUS_THRESHOLD:
        glColor4f(0.09, 0.11, 0.14, 1.0)
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(0.0, 0.0)
        for i in range(segs + 1):
            a = 2.0 * math.pi * i / segs
            glVertex2f(r_min * math.cos(a), r_min * math.sin(a))
        glEnd()
    # Outline rings using raw GL lines
    glColor4f(0.27, 0.51, 0.71, 0.5)
    glLineWidth(1.0)
    glBegin(GL_LINE_STRIP)
    for i in range(segs + 1):
        a = 2.0 * math.pi * i / segs
        glVertex2f(r_max * math.cos(a), r_max * math.sin(a))
    glEnd()
    if r_min > _INNER_RADIUS_THRESHOLD:
        glBegin(GL_LINE_STRIP)
        for i in range(segs + 1):
            a = 2.0 * math.pi * i / segs
            glVertex2f(r_min * math.cos(a), r_min * math.sin(a))
        glEnd()


def _draw_obstacle(obstacle: list[float]) -> None:
    """Draw the rectangular obstacle as a filled quad.

    Args:
        obstacle: ``[x_min, y_min, x_max, y_max]`` in metres.
    """
    xmin, ymin, xmax, ymax = obstacle
    glColor4f(0.85, 0.25, 0.25, 0.7)
    glBegin(GL_QUADS)
    glVertex2f(xmin, ymin)
    glVertex2f(xmax, ymin)
    glVertex2f(xmax, ymax)
    glVertex2f(xmin, ymax)
    glEnd()
    # Border
    glColor4f(1.0, 0.2, 0.2, 1.0)
    glLineWidth(1.5)
    glBegin(GL_LINE_LOOP)
    glVertex2f(xmin, ymin)
    glVertex2f(xmax, ymin)
    glVertex2f(xmax, ymax)
    glVertex2f(xmin, ymax)
    glEnd()


def _draw_arm(
    robot: object,
    q1: float,
    q2: float,
    color: tuple[float, float, float],
    alpha: float = 1.0,
    link_width: float = 4.0,
    joint_r: float = 0.04,
) -> None:
    """Draw the two-link arm at the given joint configuration.

    Args:
        robot: :class:`~arco.kinematics.RRRobot` instance.
        q1: First joint angle in radians.
        q2: Second joint angle in radians.
        color: ``(r, g, b)`` in ``[0, 1]``.
        alpha: Opacity in ``[0, 1]``.
        link_width: GL line width in pixels.
        joint_r: Radius of joint circles in metres.
    """
    origin, j2, ee = robot.link_segments(q1, q2)
    r, g, b = color
    glColor4f(r, g, b, alpha)
    glLineWidth(link_width)
    glBegin(GL_LINE_STRIP)
    glVertex2f(origin[0], origin[1])
    glVertex2f(j2[0], j2[1])
    glVertex2f(ee[0], ee[1])
    glEnd()
    # Joints
    for cx, cy in (origin, j2, ee):
        renderer_gl.draw_disc(cx, cy, joint_r, r, g, b)


def _draw_trail(
    trail: list[tuple[float, float]],
    color: tuple[float, float, float],
    alpha: float = 0.7,
) -> None:
    """Draw the end-effector trail as a polyline.

    Args:
        trail: List of ``(x, y)`` end-effector positions.
        color: ``(r, g, b)`` in ``[0, 1]``.
        alpha: Opacity in ``[0, 1]``.
    """
    if len(trail) < 2:
        return
    r, g, b = color
    glColor4f(r, g, b, alpha)
    glLineWidth(1.5)
    glBegin(GL_LINE_STRIP)
    for x, y in trail:
        glVertex2f(x, y)
    glEnd()


# ---------------------------------------------------------------------------
# Right-panel (joint space) drawing helpers
# ---------------------------------------------------------------------------


def _draw_cspace_scatter(
    collision_pts: list[list[float]], max_pts: int = 5000
) -> None:
    """Draw the C-space collision region as a point scatter.

    Args:
        collision_pts: List of ``[q1, q2]`` collision configurations.
        max_pts: Maximum number of points to draw for performance.
    """
    if not collision_pts:
        return
    pts = collision_pts[:: max(1, len(collision_pts) // max_pts)]
    glPointSize(2.0)
    glColor4f(0.5, 0.5, 0.5, 0.4)
    glBegin(GL_POINTS)
    for p in pts:
        glVertex2f(float(p[0]), float(p[1]))
    glEnd()


def _draw_joint_path(
    path: list[np.ndarray] | None,
    color: tuple[float, float, float],
    alpha: float = 0.8,
    line_width: float = 1.5,
) -> None:
    """Draw a joint-space path as a polyline in (theta1, theta2) space.

    Args:
        path: List of 2-element joint-config arrays, or ``None``.
        color: ``(r, g, b)`` in ``[0, 1]``.
        alpha: Opacity in ``[0, 1]``.
        line_width: GL line width in pixels.
    """
    if path is None or len(path) < 2:
        return
    r, g, b = color
    glColor4f(r, g, b, alpha)
    glLineWidth(line_width)
    glBegin(GL_LINE_STRIP)
    for pt in path:
        glVertex2f(float(pt[0]), float(pt[1]))
    glEnd()


# ---------------------------------------------------------------------------
# Main simulation function
# ---------------------------------------------------------------------------


def run_rr_sim(
    scene: RRScene,
    *,
    fps: int = 30,
    dt: float = 0.05,
    record: str = "",
    record_duration: float = 90.0,
    auto_close: bool = False,
) -> None:
    """Run the RR arm split-screen simulation.

    Args:
        scene: Pre-built :class:`RRScene` instance.
        fps: Target frames per second.
        dt: Simulation time step in seconds.
        record: If non-empty, record to this MP4 path.
        record_duration: Maximum recording duration in seconds.
        auto_close: Close the window when the simulation finishes.
    """
    recording = bool(record)
    pygame.init()
    flags = pygame.OPENGL | pygame.DOUBLEBUF
    screen = pygame.display.set_mode((_SW, _SH), flags)
    pygame.display.set_caption("ARCO — RR arm planner")
    clock = pygame.time.Clock()
    sw, sh = _SW, _SH
    left_w = int(sw * _LEFT_FRAC)

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glDisable(GL_DEPTH_TEST)
    glDisable(GL_LIGHTING)

    robot = scene.robot
    obstacle = scene.obstacle
    start_q = scene.start_q
    goal_q = scene.goal_q
    r_min, r_max = robot.workspace_annulus()
    race_speed = float(load_config("rr").get("race_speed", 0.8))

    # Build trajectory lists — prefer optimized, fall back to raw path
    def _prefer_optimized_trajectory(
        traj: list[np.ndarray], path: list[np.ndarray] | None
    ) -> list[np.ndarray] | None:
        if traj:
            return traj
        return path

    rrt_traj = _prefer_optimized_trajectory(scene.rrt_traj, scene.rrt_path)
    sst_traj = _prefer_optimized_trajectory(scene.sst_traj, scene.sst_path)

    # Precompute cumulative arc lengths
    rrt_cum: list[float] = []
    sst_cum: list[float] = []
    rrt_total_len = 0.0
    sst_total_len = 0.0
    if rrt_traj and len(rrt_traj) >= 2:
        rrt_cum = _cumulative_lengths(rrt_traj)
        rrt_total_len = rrt_cum[-1]
    if sst_traj and len(sst_traj) >= 2:
        sst_cum = _cumulative_lengths(sst_traj)
        sst_total_len = sst_cum[-1]

    # FK shortcuts
    def _fk(q: np.ndarray) -> tuple[float, float]:
        return robot.forward_kinematics(float(q[0]), float(q[1]))

    # Start / goal Cartesian positions
    start_cart = _fk(start_q)
    goal_cart = _fk(goal_q)

    # Simulation state
    sim_time = 0.0
    paused = False
    done = False
    post_finish_timer = 0.0

    # Which planner is currently being shown ("rrt" or "sst")
    current_planner = "rrt"
    current_traj = rrt_traj
    current_cum = rrt_cum
    current_total = rrt_total_len
    progress_s = 0.0

    # End-effector trail
    ee_trail: list[tuple[float, float]] = []

    # Joint-space trace (recorded q1, q2 as the arm moves)
    joint_trace: list[tuple[float, float]] = []

    # Pre-draw static C-space elements into a GL display list would be ideal
    # but for simplicity we redraw each frame.
    collision_pts = scene.collision_pts

    # Colors
    rrt_color = _cf(_C_RRT_TRAJ)
    sst_color = _cf(_C_SST_TRAJ)

    video_writer: VideoWriter | None = None
    record_frames = 0
    max_record_frames = int(record_duration * fps) if recording else 0

    if recording:
        video_writer = VideoWriter(record, sw, sh, fps)
        video_writer.__enter__()
        auto_close = True
        logger.info("Recording to %s (%.0f s max)", record, record_duration)

    # Text rendering (pygame font → GL texture)
    pygame.font.init()
    font = pygame.font.SysFont("monospace", 16)

    def _draw_label_gl(
        text: str,
        x_px: int,
        y_px: int,
        color: tuple[int, int, int] = (200, 200, 200),
    ) -> None:
        surf = font.render(text, True, color)
        renderer_gl.blit_overlay(surf, x_px, y_px, sw, sh)

    while True:
        # ---- Events -------------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    done = True
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    # Restart
                    sim_time = 0.0
                    progress_s = 0.0
                    current_planner = "rrt"
                    current_traj = rrt_traj
                    current_cum = rrt_cum
                    current_total = rrt_total_len
                    ee_trail.clear()
                    joint_trace.clear()
                    post_finish_timer = 0.0
        if done:
            break

        # ---- Recording stop -----------------------------------------------
        if recording and record_frames >= max_record_frames:
            break

        # ---- Simulation step ----------------------------------------------
        if not paused:
            sim_time += dt
            progress_s = min(progress_s + race_speed * dt, current_total)

            # Current joint config
            if current_traj and current_cum:
                q_now = _interp_path(current_traj, current_cum, progress_s)
            else:
                q_now = goal_q.copy()
                progress_s = current_total

            ee_now = _fk(q_now)
            ee_trail.append(ee_now)
            joint_trace.append((float(q_now[0]), float(q_now[1])))

            # Phase transitions
            if progress_s >= current_total:
                post_finish_timer += dt
                if post_finish_timer >= _POST_FINISH_SECS:
                    post_finish_timer = 0.0
                    if current_planner == "rrt" and sst_traj is not None:
                        current_planner = "sst"
                        current_traj = sst_traj
                        current_cum = sst_cum
                        current_total = sst_total_len
                        progress_s = 0.0
                        ee_trail.clear()
                        joint_trace.clear()
                        logger.info("Switching to SST tracking.")
                    elif current_planner == "sst":
                        if auto_close:
                            done = True
                        # Otherwise keep going — loop SST
                        progress_s = 0.0
                        ee_trail.clear()
                        joint_trace.clear()

        # ---- Render -------------------------------------------------------
        bg = _cf(_C_BG)
        glClearColor(bg[0], bg[1], bg[2], 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        # --- LEFT panel: Cartesian workspace ---
        _set_left_viewport(sw, sh, left_w)

        _draw_workspace_annulus(r_min, r_max)
        _draw_obstacle(obstacle)

        # Planned path trace (dim)
        if current_planner == "rrt" and scene.rrt_path:
            fk_path = [_fk(p) for p in scene.rrt_path]
            color = _cf(_C_RRT_PATH)
            glColor4f(color[0], color[1], color[2], 0.3)
            glLineWidth(1.0)
            glBegin(GL_LINE_STRIP)
            for x, y in fk_path:
                glVertex2f(x, y)
            glEnd()
        elif current_planner == "sst" and scene.sst_path:
            fk_path = [_fk(p) for p in scene.sst_path]
            color = _cf(_C_SST_PATH)
            glColor4f(color[0], color[1], color[2], 0.3)
            glLineWidth(1.0)
            glBegin(GL_LINE_STRIP)
            for x, y in fk_path:
                glVertex2f(x, y)
            glEnd()

        # End-effector trail
        if current_planner == "rrt":
            _draw_trail(ee_trail, rrt_color, alpha=0.7)
        else:
            _draw_trail(ee_trail, sst_color, alpha=0.7)

        # Robot arm
        arm_color = rrt_color if current_planner == "rrt" else sst_color
        if current_traj and current_cum:
            q_draw = _interp_path(current_traj, current_cum, progress_s)
        else:
            q_draw = goal_q
        _draw_arm(robot, float(q_draw[0]), float(q_draw[1]), arm_color)

        # Start / goal markers
        renderer_gl.draw_disc(
            start_cart[0], start_cart[1], 0.05, 0.196, 0.804, 0.196
        )
        renderer_gl.draw_disc(
            goal_cart[0], goal_cart[1], 0.06, 1.0, 0.314, 0.118
        )

        # --- RIGHT panel: joint C-space ---
        _set_right_viewport(sw, sh, left_w)

        _draw_cspace_scatter(collision_pts)

        # All paths (dim) in joint space
        if scene.rrt_path:
            _draw_joint_path(
                scene.rrt_path, _cf(_C_RRT_PATH), alpha=0.3, line_width=1.0
            )
        if scene.sst_path:
            _draw_joint_path(
                scene.sst_path, _cf(_C_SST_PATH), alpha=0.3, line_width=1.0
            )

        # Optimized trajectories (bright)
        if rrt_traj:
            _draw_joint_path(rrt_traj, rrt_color, alpha=0.6, line_width=1.5)
        if sst_traj:
            _draw_joint_path(sst_traj, sst_color, alpha=0.6, line_width=1.5)

        # Joint trace of current planner
        if len(joint_trace) >= 2:
            tc = rrt_color if current_planner == "rrt" else sst_color
            glColor4f(tc[0], tc[1], tc[2], 0.9)
            glLineWidth(2.5)
            glBegin(GL_LINE_STRIP)
            for tq1, tq2 in joint_trace:
                glVertex2f(tq1, tq2)
            glEnd()

        # Live dot (current config)
        live_q1 = float(q_draw[0])
        live_q2 = float(q_draw[1])
        renderer_gl.draw_disc(live_q1, live_q2, 0.08, 1.0, 0.863, 0.196)

        # Start / goal markers in C-space
        renderer_gl.draw_disc(
            float(start_q[0]), float(start_q[1]), 0.07, 0.196, 0.804, 0.196
        )
        renderer_gl.draw_disc(
            float(goal_q[0]), float(goal_q[1]), 0.09, 1.0, 0.314, 0.118
        )

        # Axes (draw short lines at ±π)
        Q = math.pi
        glColor4f(0.4, 0.4, 0.4, 0.8)
        glLineWidth(1.0)
        glBegin(GL_LINES)
        glVertex2f(-Q, 0.0)
        glVertex2f(Q, 0.0)
        glVertex2f(0.0, -Q)
        glVertex2f(0.0, Q)
        glEnd()

        # --- Full-screen HUD overlay ---
        _reset_viewport(sw, sh)

        label = "RRT*" if current_planner == "rrt" else "SST"
        pct = (
            100.0 * progress_s / current_total
            if current_total > 1e-9
            else 100.0
        )
        _draw_label_gl(
            f"{label}  {pct:5.1f}%  t={sim_time:.1f}s",
            10,
            sh - 24,
            color=(220, 220, 220),
        )
        _draw_label_gl(
            "SPACE=pause  R=restart  Q=quit",
            10,
            sh - 44,
            color=(140, 140, 140),
        )
        # Panel labels
        _draw_label_gl(
            "Cartesian workspace",
            10,
            sh - 64,
            color=(100, 160, 220),
        )
        _draw_label_gl(
            "Joint C-space  (\u03b81, \u03b82)",
            left_w + 10,
            sh - 24,
            color=(100, 200, 140),
        )

        pygame.display.flip()

        if recording and video_writer is not None:
            video_writer.write_frame_gl()
            record_frames += 1

        clock.tick(fps)

    # Cleanup
    if video_writer is not None:
        video_writer.__exit__(None, None, None)
        logger.info("Video saved.")
    pygame.quit()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments, build the scene, and launch the simulator."""
    configure_logging()
    parser = argparse.ArgumentParser(
        description="RR arm split-screen simulator"
    )
    parser.add_argument("--fps", type=int, default=30, help="Target FPS.")
    parser.add_argument(
        "--record",
        metavar="PATH",
        default="",
        help="Record simulation to this MP4 file.",
    )
    parser.add_argument(
        "--record-duration",
        type=float,
        default=90.0,
        metavar="SECS",
        help="Maximum recording duration in seconds (default: 90).",
    )
    parser.add_argument(
        "--close",
        action="store_true",
        help="Close the window automatically when the simulation ends.",
    )
    args = parser.parse_args()

    cfg = load_config("rr")
    scene = RRScene(cfg)

    pygame.init()
    flags = pygame.OPENGL | pygame.DOUBLEBUF
    pygame.display.set_mode((_SW, _SH), flags)
    pygame.display.set_caption("ARCO — RR arm (loading…)")

    run_with_loading_screen(scene, _SW, _SH)

    pygame.display.set_caption("ARCO — RR arm planner")

    run_rr_sim(
        scene,
        fps=args.fps,
        record=args.record,
        record_duration=args.record_duration,
        auto_close=args.close or bool(args.record),
    )


if __name__ == "__main__":
    main()
