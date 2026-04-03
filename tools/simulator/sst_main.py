#!/usr/bin/env python
"""Pygame visualisation of an SST planner on a 2-D obstacle field.

Runs the Sparse Stable Trees (SST) algorithm to completion, then
animates the active exploration tree growing incrementally at a fixed
frame rate.  When the tree is fully revealed the optimal path is
highlighted.

Keyboard controls
-----------------
Q / Escape  Quit

Usage
-----
Interactive window::

    python tools/simulator/sst_main.py

Headless MP4 recording::

    python tools/simulator/sst_main.py \\
        --fps 30 \\
        --record /tmp/sst_planning.mp4 \\
        --record-duration 60
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, ".."))

import pygame
from renderer import WorldTransform

from arco.mapping import KDTreeOccupancy
from arco.planning.continuous import SSTPlanner
from config import load_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C_BG = (28, 28, 35)
C_OBSTACLE = (160, 60, 60)
C_TREE_EDGE = (60, 160, 140)
C_TREE_NODE = (50, 140, 120)
C_PATH = (230, 170, 30)
C_START = (60, 200, 90)
C_GOAL = (220, 80, 220)
C_HUD = (220, 220, 220)
C_HUD_SHADOW = (40, 40, 50)

_DEFAULT_SCREEN_W = 1280
_DEFAULT_SCREEN_H = 800

_cfg = load_config("sst")


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
    """Draw the first *count* active tree nodes and their parent edges.

    Args:
        surface: Target pygame surface.
        nodes: Active tree nodes returned by the SST planner.
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
        revealed: Number of active nodes currently visible.
        total: Total number of active nodes.
        path_found: Whether a solution path exists.
    """
    lines = [
        f"Nodes: {revealed}/{total}",
        f"Path: {'found' if path_found else 'none'}",
        "SST",
    ]
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
) -> None:
    """Run the pygame SST visualisation.

    Args:
        fps: Target frame rate (frames per second).
        record: If non-empty, render headlessly and save an MP4 to this path.
        record_duration: Maximum recording duration in seconds.
    """
    recording = bool(record)
    max_record_frames = int(fps * record_duration)

    if recording:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    pygame.init()
    screen_w, screen_h = _DEFAULT_SCREEN_W, _DEFAULT_SCREEN_H
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("SST Planning Visualisation")
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

    planner = SSTPlanner(
        occ,
        bounds=bounds,
        max_sample_count=int(_cfg["max_sample_count"]),
        step_size=float(_cfg["step_size"]),
        goal_tolerance=float(_cfg["goal_tolerance"]),
        witness_radius=float(_cfg["witness_radius"]),
        collision_check_count=int(_cfg["collision_check_count"]),
        goal_bias=float(_cfg["goal_bias"]),
    )

    logger.info("Running SST …")
    tree_nodes, tree_parent, path = planner.get_tree(start, goal)
    total_nodes = len(tree_nodes)
    logger.info(
        "Active tree complete: %d nodes, path=%s",
        total_nodes,
        "found" if path is not None else "none",
    )

    # World-to-screen transform fitted to the planning bounds
    corner_pts: List[Tuple[float, float]] = [
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

    revealed = 0
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

        # Reveal more nodes each frame until all are shown
        if revealed < total_nodes:
            revealed = min(revealed + nodes_per_frame, total_nodes)

        # Render
        screen.fill(C_BG)
        _draw_obstacles(screen, occ, transform)
        _draw_tree(screen, tree_nodes, tree_parent, revealed, transform)
        if revealed >= total_nodes and path is not None:
            _draw_path(screen, path, transform)
        _draw_endpoints(screen, start, goal, transform)
        _draw_hud(
            screen,
            font,
            revealed,
            total_nodes,
            path is not None,
        )

        if recording:
            _write_video_frame(video_writer, screen)
            record_frames += 1
            if record_frames >= max_record_frames:
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
    args = parser.parse_args()
    main(
        fps=args.fps, record=args.record, record_duration=args.record_duration
    )
