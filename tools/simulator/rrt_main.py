#!/usr/bin/env python
"""Pygame visualisation of an RRT* planner on a 2-D obstacle field.

Runs the asymptotically-optimal RRT* algorithm to completion, then
animates the exploration tree growing incrementally at a fixed frame rate.
When the tree is fully revealed the optimal path is highlighted.

Keyboard controls
-----------------
Q / Escape  Quit

Usage
-----
Interactive window::

    python tools/simulator/rrt_main.py

Headless MP4 recording::

    python tools/simulator/rrt_main.py \\
        --fps 30 \\
        --record /tmp/rrt_planning.mp4 \\
        --record-duration 60

Zoomed-in recording (path bounding box)::

    python tools/simulator/rrt_main.py \\
        --fps 30 \\
        --zoom \\
        --record /tmp/rrt_zoom.mp4 \\
        --record-duration 60
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, ".."))

import pygame
from renderer import (
    WorldTransform,
    draw_tracking_target,
    draw_trajectory,
    draw_vehicle,
)

from arco.guidance.control.pure_pursuit import PurePursuitController
from arco.guidance.control.tracking import TrackingLoop
from arco.guidance.vehicle import DubinsVehicle
from arco.mapping import KDTreeOccupancy
from arco.planning.continuous import RRTPlanner
from config import load_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C_BG = (28, 28, 35)
C_OBSTACLE = (160, 60, 60)
C_TREE_EDGE = (60, 120, 180)
C_TREE_NODE = (50, 100, 160)
C_PATH = (230, 170, 30)
C_START = (60, 200, 90)
C_GOAL = (220, 80, 220)
C_HUD = (220, 220, 220)
C_HUD_SHADOW = (40, 40, 50)

_DEFAULT_SCREEN_W = 1280
_DEFAULT_SCREEN_H = 800

_cfg = load_config("rrt")

# ---------------------------------------------------------------------------
# Vehicle parameters scaled for the 50 x 50 m planning environment
# ---------------------------------------------------------------------------
_VEH_MAX_SPEED = 5.0
_VEH_CRUISE_SPEED = 3.0
_VEH_LOOKAHEAD = 4.0
_VEH_GOAL_RADIUS = 3.0
_VEH_MAX_TURN_RATE = math.radians(90.0)
_VEH_MAX_ACCEL = 4.9
_VEH_MAX_TURN_RATE_DOT = math.radians(3600.0)
_VEH_DT = 1.0 / 30.0

# Frames to hold the completed tree before transitioning to the vehicle phase
_HOLD_FRAMES = 60


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def _build_occupancy() -> KDTreeOccupancy:
    """Build the scattered-obstacle environment used for demonstration.

    Returns:
        A :class:`~arco.mapping.KDTreeOccupancy` with a central wall and
        scattered point obstacles (deterministic seed).
    """
    rng = np.random.default_rng(7)
    x_max = float(_cfg["bounds"][0][1])
    y_max = float(_cfg["bounds"][1][1])

    wall_pts = [[x, y_max / 2.0] for x in np.arange(0.0, x_max * 0.6, 1.5)] + [
        [x, y_max / 2.0] for x in np.arange(x_max * 0.7, x_max, 1.5)
    ]

    margin = 5.0
    scatter_count = 40
    scatter_pts: list[list[float]] = []
    while len(scatter_pts) < scatter_count:
        p = rng.uniform([margin, margin], [x_max - margin, y_max - margin])
        scatter_pts.append(p.tolist())

    all_pts = wall_pts + scatter_pts
    return KDTreeOccupancy(
        all_pts, clearance=float(_cfg["obstacle_clearance"])
    )


# ---------------------------------------------------------------------------
# Vehicle simulation helpers
# ---------------------------------------------------------------------------


def _initial_heading(path: List[np.ndarray]) -> float:
    """Return heading (radians) from path[0] toward path[1].

    Args:
        path: Ordered list of numpy waypoints.

    Returns:
        Heading angle in radians, or 0.0 if fewer than 2 points.
    """
    if len(path) < 2:
        return 0.0
    dx = float(path[1][0]) - float(path[0][0])
    dy = float(path[1][1]) - float(path[0][1])
    return math.atan2(dy, dx)


def _find_lookahead(
    x: float,
    y: float,
    path: List[Tuple[float, float]],
    lookahead: float,
) -> Tuple[float, float]:
    """Return the lookahead point on *path* at least *lookahead* metres away.

    Args:
        x: Current x-position in world metres.
        y: Current y-position in world metres.
        path: Ordered list of ``(x, y)`` waypoints.
        lookahead: Minimum distance to the lookahead target in metres.

    Returns:
        ``(x, y)`` of the lookahead target.
    """
    if not path:
        return (x, y)
    closest = min(
        range(len(path)),
        key=lambda i: math.hypot(path[i][0] - x, path[i][1] - y),
    )
    for pt in path[closest:]:
        if math.hypot(pt[0] - x, pt[1] - y) >= lookahead:
            return pt
    return path[-1]


def _build_vehicle_sim(
    path: List[np.ndarray],
) -> Tuple[DubinsVehicle, TrackingLoop, List[Tuple[float, float]]]:
    """Create a Dubins vehicle and tracking loop initialised at path start.

    Args:
        path: Planned path as ordered numpy waypoints.

    Returns:
        Tuple of ``(vehicle, tracking_loop, smooth_path)`` where
        *smooth_path* is the path converted to ``(x, y)`` tuples.
    """
    smooth_path = [(float(p[0]), float(p[1])) for p in path]
    x0, y0 = smooth_path[0]
    theta0 = _initial_heading(path)
    vehicle = DubinsVehicle(
        x=x0,
        y=y0,
        heading=theta0,
        max_speed=_VEH_MAX_SPEED,
        min_speed=0.0,
        max_turn_rate=_VEH_MAX_TURN_RATE,
        max_acceleration=_VEH_MAX_ACCEL,
        max_turn_rate_dot=_VEH_MAX_TURN_RATE_DOT,
    )
    controller = PurePursuitController(lookahead_distance=_VEH_LOOKAHEAD)
    loop = TrackingLoop(
        vehicle,
        controller,
        cruise_speed=_VEH_CRUISE_SPEED,
        curvature_gain=0.0,
    )
    return vehicle, loop, smooth_path


# ---------------------------------------------------------------------------
# Video helpers (mirror of main.py)
# ---------------------------------------------------------------------------


def _open_video_writer(
    path: str, width: int, height: int, fps: int
) -> "subprocess.Popen[bytes]":
    """Open an ffmpeg subprocess that reads raw RGB frames from stdin.

    Args:
        path: Output MP4 file path.
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Frames per second.

    Returns:
        Running :class:`subprocess.Popen` instance with an open ``stdin``.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "23",
        path,
    ]
    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _write_video_frame(
    proc: "subprocess.Popen[bytes]", surface: "pygame.Surface"
) -> None:
    """Capture the current pygame surface and write it to the ffmpeg pipe.

    Args:
        proc: ffmpeg subprocess with an open ``stdin`` pipe.
        surface: Pygame surface to capture.
    """
    if proc.stdin is None:
        raise RuntimeError("ffmpeg stdin pipe is not open")
    frame = pygame.surfarray.array3d(surface)
    frame = np.ascontiguousarray(frame.transpose(1, 0, 2))
    proc.stdin.write(frame.tobytes())


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def _draw_obstacles(
    surface: pygame.Surface,
    occ: KDTreeOccupancy,
    transform: WorldTransform,
) -> None:
    """Draw each obstacle point as a small filled circle.

    Args:
        surface: Target pygame surface.
        occ: Occupancy map whose obstacle points are drawn.
        transform: World-to-screen coordinate transform.
    """
    for pt in occ.points:
        sx, sy = transform(float(pt[0]), float(pt[1]))
        pygame.draw.circle(surface, C_OBSTACLE, (sx, sy), 4)


def _draw_tree(
    surface: pygame.Surface,
    nodes: List[np.ndarray],
    parent: Dict[int, Optional[int]],
    count: int,
    transform: WorldTransform,
) -> None:
    """Draw the first *count* tree nodes and their parent edges.

    Args:
        surface: Target pygame surface.
        nodes: All tree nodes (full list from planner).
        parent: Mapping from node index to parent index (``None`` for root).
        count: Number of nodes to draw (from index 0).
        transform: World-to-screen coordinate transform.
    """
    for i in range(min(count, len(nodes))):
        sx, sy = transform(float(nodes[i][0]), float(nodes[i][1]))
        p = parent.get(i)
        if p is not None:
            px, py = transform(float(nodes[p][0]), float(nodes[p][1]))
            pygame.draw.line(surface, C_TREE_EDGE, (px, py), (sx, sy), 1)
        pygame.draw.circle(surface, C_TREE_NODE, (sx, sy), 2)


def _draw_path(
    surface: pygame.Surface,
    path: List[np.ndarray],
    transform: WorldTransform,
) -> None:
    """Draw the solution path as a thick yellow polyline.

    Args:
        surface: Target pygame surface.
        path: Ordered list of numpy waypoints.
        transform: World-to-screen coordinate transform.
    """
    if len(path) < 2:
        return
    pts = [transform(float(p[0]), float(p[1])) for p in path]
    pygame.draw.lines(surface, C_PATH, False, pts, 3)


def _draw_endpoints(
    surface: pygame.Surface,
    start: np.ndarray,
    goal: np.ndarray,
    transform: WorldTransform,
) -> None:
    """Draw start (green) and goal (magenta) markers.

    Args:
        surface: Target pygame surface.
        start: Start position in world coordinates.
        goal: Goal position in world coordinates.
        transform: World-to-screen coordinate transform.
    """
    sx, sy = transform(float(start[0]), float(start[1]))
    gx, gy = transform(float(goal[0]), float(goal[1]))
    pygame.draw.circle(surface, C_START, (sx, sy), 8)
    pygame.draw.circle(surface, C_BG, (sx, sy), 4)
    pygame.draw.circle(surface, C_GOAL, (gx, gy), 8)
    pygame.draw.circle(surface, C_BG, (gx, gy), 4)


def _draw_hud(
    surface: pygame.Surface,
    font: "pygame.font.Font",
    revealed: int,
    total: int,
    path_found: bool,
) -> None:
    """Draw a minimal HUD showing planning progress.

    Args:
        surface: Target pygame surface.
        font: Pygame font for rendering text.
        revealed: Number of tree nodes currently visible.
        total: Total number of tree nodes.
        path_found: Whether a solution path exists.
    """
    lines = [
        f"Nodes: {revealed}/{total}",
        f"Path: {'found' if path_found else 'none'}",
        "RRT*",
    ]
    x, y = 10, 10
    for line in lines:
        shadow = font.render(line, True, C_HUD_SHADOW)
        surface.blit(shadow, (x + 1, y + 1))
        text = font.render(line, True, C_HUD)
        surface.blit(text, (x, y))
        y += font.get_linesize() + 2


def _draw_vehicle_hud(
    surface: pygame.Surface,
    font: "pygame.font.Font",
    step: int,
    speed: float,
    cte: float,
    finished: bool,
) -> None:
    """Draw a HUD showing Dubins vehicle tracking state.

    Args:
        surface: Target pygame surface.
        font: Pygame font for rendering text.
        step: Current simulation step count.
        speed: Current vehicle speed in m/s.
        cte: Cross-track error in metres.
        finished: Whether the vehicle has reached the goal.
    """
    lines = [
        f"Step: {step}",
        f"Speed: {speed:.1f} m/s",
        f"CTE: {cte:+.1f} m",
        "RRT* — execution",
    ]
    if finished:
        lines.append("[ GOAL REACHED ]")
    x, y = 10, 10
    for line in lines:
        shadow = font.render(line, True, C_HUD_SHADOW)
        surface.blit(shadow, (x + 1, y + 1))
        text = font.render(line, True, C_HUD)
        surface.blit(text, (x, y))
        y += font.get_linesize() + 2


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(
    fps: int = 30,
    record: str = "",
    record_duration: float = 60.0,
    zoom: bool = False,
) -> None:
    """Run the pygame RRT* visualisation.

    Args:
        fps: Target frame rate (frames per second).
        record: If non-empty, render headlessly and save an MP4 to this path.
        record_duration: Maximum recording duration in seconds.
        zoom: If True, fit the camera to the path bounding box instead of
            the full planning bounds.
    """
    recording = bool(record)
    max_record_frames = int(fps * record_duration)

    if recording:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    pygame.init()
    screen_w, screen_h = _DEFAULT_SCREEN_W, _DEFAULT_SCREEN_H
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("RRT* Planning Visualisation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)

    # Build environment and run planner once
    occ = _build_occupancy()
    bounds = [tuple(b) for b in _cfg["bounds"]]
    start = np.array([2.0, 2.0])
    goal = np.array(
        [
            float(_cfg["bounds"][0][1]) - 2.0,
            float(_cfg["bounds"][1][1]) - 2.0,
        ]
    )

    planner = RRTPlanner(
        occ,
        bounds=bounds,
        max_sample_count=int(_cfg["max_sample_count"]),
        step_size=float(_cfg["step_size"]),
        goal_tolerance=float(_cfg["goal_tolerance"]),
        collision_check_count=int(_cfg["collision_check_count"]),
        goal_bias=float(_cfg["goal_bias"]),
    )

    logger.info("Running RRT* …")
    tree_nodes, tree_parent, path = planner.get_tree(start, goal)
    total_nodes = len(tree_nodes)
    logger.info(
        "Tree complete: %d nodes, path=%s",
        total_nodes,
        "found" if path is not None else "none",
    )

    # World-to-screen transform fitted to the planning bounds (or path bbox)
    if zoom and path is not None:
        path_xs = [float(p[0]) for p in path]
        path_ys = [float(p[1]) for p in path]
        pad = max(
            (max(path_xs) - min(path_xs)) * 0.15,
            (max(path_ys) - min(path_ys)) * 0.15,
            5.0,
        )
        corner_pts = [
            (min(path_xs) - pad, min(path_ys) - pad),
            (max(path_xs) + pad, max(path_ys) + pad),
        ]
    else:
        corner_pts = [
            (float(bounds[0][0]), float(bounds[1][0])),
            (float(bounds[0][1]), float(bounds[1][1])),
        ]
    transform = WorldTransform(corner_pts, (screen_w, screen_h), margin=60)

    # Compute how many nodes to reveal per frame so the animation fills
    # roughly half the recording duration, then holds the final frame.
    half_frames = max(1, max_record_frames // 2)
    nodes_per_frame = max(1, total_nodes // half_frames)

    video_writer = None
    if recording:
        video_writer = _open_video_writer(record, screen_w, screen_h, fps)
        logger.info(
            "Recording to %r (%dx%d @ %d fps)", record, screen_w, screen_h, fps
        )

    # Two-phase state: "tree" → exploration animation; "vehicle" → execution
    phase = "tree"
    revealed = 0
    hold = 0  # frames held after tree is complete before vehicle phase

    # Vehicle phase state (initialised when phase transitions)
    vehicle: Optional[DubinsVehicle] = None
    veh_loop: Optional[TrackingLoop] = None
    smooth_path: List[Tuple[float, float]] = []
    veh_trajectory: List[Tuple[float, float, float]] = []
    veh_finished = False
    veh_step = 0

    record_frames = 0
    running = True

    while running:
        if not recording:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False

        # ----------------------------------------------------------------
        # Phase logic
        # ----------------------------------------------------------------
        if phase == "tree":
            if revealed < total_nodes:
                revealed = min(revealed + nodes_per_frame, total_nodes)
            else:
                hold += 1
                # Transition to vehicle phase after hold or at half-way point
                transition = hold >= _HOLD_FRAMES or (
                    recording and record_frames >= half_frames
                )
                if transition and path is not None:
                    vehicle, veh_loop, smooth_path = _build_vehicle_sim(path)
                    veh_trajectory = []
                    veh_finished = False
                    veh_step = 0
                    phase = "vehicle"
                    logger.info("Switching to vehicle execution phase.")
        elif phase == "vehicle":
            assert vehicle is not None and veh_loop is not None
            if not veh_finished:
                veh_loop.step(smooth_path, dt=_VEH_DT)
                veh_step += 1
                veh_trajectory.append(vehicle.pose)
                gx, gy = smooth_path[-1]
                if (
                    math.hypot(vehicle.x - gx, vehicle.y - gy)
                    < _VEH_GOAL_RADIUS
                ):
                    veh_finished = True
                    logger.info("Vehicle reached goal in %d steps.", veh_step)

        # ----------------------------------------------------------------
        # Render
        # ----------------------------------------------------------------
        screen.fill(C_BG)
        _draw_obstacles(screen, occ, transform)

        if phase == "tree":
            _draw_tree(screen, tree_nodes, tree_parent, revealed, transform)
            if revealed >= total_nodes and path is not None:
                _draw_path(screen, path, transform)
            _draw_endpoints(screen, start, goal, transform)
            _draw_hud(screen, font, revealed, total_nodes, path is not None)
        else:
            # Show full tree and path as background context
            _draw_tree(screen, tree_nodes, tree_parent, total_nodes, transform)
            if path is not None:
                _draw_path(screen, path, transform)
            _draw_endpoints(screen, start, goal, transform)
            assert vehicle is not None
            if len(veh_trajectory) >= 2:
                draw_trajectory(screen, veh_trajectory, transform)
            la = _find_lookahead(
                vehicle.x, vehicle.y, smooth_path, _VEH_LOOKAHEAD
            )
            draw_tracking_target(screen, la, transform)
            draw_vehicle(
                screen, vehicle.x, vehicle.y, vehicle.heading, transform
            )
            metrics = (veh_loop.metrics or {}) if veh_loop is not None else {}
            _draw_vehicle_hud(
                screen,
                font,
                veh_step,
                float(metrics.get("speed", 0.0)),
                float(metrics.get("cross_track_error", 0.0)),
                veh_finished,
            )

        if recording:
            _write_video_frame(video_writer, screen)
            record_frames += 1
            if record_frames >= max_record_frames:
                running = False
            elif phase == "vehicle" and veh_finished:
                running = False
        else:
            pygame.display.flip()
            clock.tick(fps)

    if video_writer is not None:
        video_writer.stdin.close()
        returncode = video_writer.wait()
        if returncode != 0:
            logger.error(
                "ffmpeg exited with code %d; video may be incomplete.",
                returncode,
            )
        else:
            logger.info("Video saved to %r", record)

    pygame.quit()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        metavar="N",
        help="Target frame rate (default: 30)",
    )
    parser.add_argument(
        "--record",
        metavar="FILE",
        default="",
        help="Record a headless MP4 to FILE and exit (requires ffmpeg).",
    )
    parser.add_argument(
        "--record-duration",
        type=float,
        default=60.0,
        metavar="S",
        dest="record_duration",
        help="Maximum recording duration in seconds (default: 60).",
    )
    parser.add_argument(
        "--zoom",
        action="store_true",
        default=False,
        help="Zoom camera to the path bounding box instead of full bounds.",
    )
    args = parser.parse_args()
    main(
        fps=args.fps,
        record=args.record,
        record_duration=args.record_duration,
        zoom=args.zoom,
    )
