#!/usr/bin/env python
"""Real-time Pygame frontend for the ARCO A* horse auto-follow pipeline.

Builds a procedural road network from ``tools/config/graph.yml``, plans an A*
route, and animates the full planning + tracking loop in real time using
Pygame.

Keyboard controls
-----------------
SPACE       Pause / resume simulation
R           Restart from the beginning
C           Toggle camera mode (full view / follow vehicle)
+ / -       Zoom in / out in follow camera mode
Q / Escape  Quit

What is displayed
-----------------
- **Road network** — gray polylines with waypoint-accurate curves
- **Planned route** — highlighted edges in red/orange
- **Smooth path** — dashed orange polyline through edge waypoints
- **Past trajectory** — blue trace of vehicle history
- **Lookahead target** — yellow circle on the smooth path
- **Vehicle** — green rectangle with heading arrow

Architecture
------------
All simulation state (route planning, vehicle dynamics, path tracking) is
handled by the backend modules in ``arco.*``.  This script is a **pure
display adapter**: it only reads the backend state after each ``loop.step()``
call and passes it to :mod:`renderer` drawing functions.  The backend
remains fully deterministic and independent of the GUI framework.

Usage
-----
::

    cd tools/simulator
    pip install -r requirements.txt     # installs pygame
    python main.py

    # Or from the repo root (after installing arco):
    python tools/simulator/main.py

Optional flags::

    python main.py --fps 60     # cap frame rate (default: 30)
    python main.py --dt 0.05    # simulation timestep in seconds (default: 0.1)
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys

# Make arco and tools packages importable without a full install.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, ".."))

import pygame
from graph.generator import generate_graph
from renderer import (
    WorldTransform,
    draw_hud,
    draw_legend,
    draw_road_network,
    draw_smooth_path,
    draw_tracking_target,
    draw_trajectory,
    draw_vehicle,
)

from arco.guidance.pure_pursuit import PurePursuitController
from arco.guidance.tracking import TrackingLoop
from arco.guidance.vehicle import DubinsVehicle
from arco.planning.discrete import RouteRouter
from config import load_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_veh_cfg = load_config("vehicle")["dubins"]
_graph_cfg = load_config("graph")

_DEFAULT_SCREEN_W = 1280
_DEFAULT_SCREEN_H = 800
TITLE = (
    f"{_graph_cfg.get('type', 'ring').title()}: "
    "Path Tracking (SPACE=pause, R=restart)"
)

# Small metric offset applied to start/goal so the vehicle begins slightly
# off a graph node, exercising the projection logic in RouteRouter.
_ENDPOINT_OFFSET_M = 4.0

_ACTIVATION_RADIUS = 35.0

_FOLLOW_ZOOM_DEFAULT = 2.0
_FOLLOW_ZOOM_MIN = 0.4
_FOLLOW_ZOOM_MAX = 8.0
_FOLLOW_ZOOM_STEP = 0.2

# 2nd-order linear camera center filter parameters.
_CAMERA_DAMPING_RATIO = 1.0
_CAMERA_NATURAL_FREQUENCY = 3.0


# ---------------------------------------------------------------------------
# Simulation helpers (same logic as astar_pipeline.py)
# ---------------------------------------------------------------------------


def _build_smooth_path(graph, route):
    """Return dense smooth path through all edge waypoints."""
    if len(route) < 2:
        return [tuple(graph.position(route[0]))] if route else []
    smooth = []
    for i in range(len(route) - 1):
        pts = graph.full_edge_geometry(route[i], route[i + 1])
        smooth.extend(pts[:-1])
    smooth.append(tuple(graph.position(route[-1])))
    return smooth


def _initial_heading(path):
    """Heading from path[0] to path[1]."""
    if len(path) < 2:
        return 0.0
    dx = path[1][0] - path[0][0]
    dy = path[1][1] - path[0][1]
    return math.atan2(dy, dx)


def _find_farthest_pair(graph, node_ids):
    """Return world positions of the two farthest-apart nodes."""
    best_dist = -1.0
    best_pair = (node_ids[0], node_ids[-1])
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            xi, yi = graph.position(node_ids[i])
            xj, yj = graph.position(node_ids[j])
            d = math.hypot(xi - xj, yi - yj)
            if d > best_dist:
                best_dist = d
                best_pair = (node_ids[i], node_ids[j])
    return (
        tuple(graph.position(best_pair[0])),
        tuple(graph.position(best_pair[1])),
    )


def _find_lookahead(x, y, path, lookahead):
    """Lookahead point on *path* at least *lookahead* metres away."""
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


def _resolve_window_size() -> tuple[int, int]:
    """Return window size from system display info, or fallback defaults."""
    info = pygame.display.Info()
    width = int(getattr(info, "current_w", 0) or 0)
    height = int(getattr(info, "current_h", 0) or 0)
    if width <= 0 or height <= 0:
        logger.warning(
            "Display size unavailable; using fallback %dx%d",
            _DEFAULT_SCREEN_W,
            _DEFAULT_SCREEN_H,
        )
        return (_DEFAULT_SCREEN_W, _DEFAULT_SCREEN_H)

    # Keep a small margin from full-screen to avoid WM quirks.
    width = max(640, int(width * 0.9))
    height = max(480, int(height * 0.9))
    return (width, height)


class _FollowTransform:
    """World-to-screen transform centred on a filtered camera position."""

    def __init__(
        self,
        center_x: float,
        center_y: float,
        screen_size: tuple[int, int],
        scale: float,
    ) -> None:
        self._center_x = center_x
        self._center_y = center_y
        self._screen_w, self._screen_h = screen_size
        self._scale = scale

    def __call__(self, wx: float, wy: float) -> tuple[int, int]:
        sx = int(self._screen_w / 2 + (wx - self._center_x) * self._scale)
        sy = int(self._screen_h / 2 - (wy - self._center_y) * self._scale)
        return (sx, sy)

    @property
    def scale(self) -> float:
        return self._scale


def build_simulation(graph):
    """Build and return all simulation objects for one run.

    Args:
        graph: Pre-loaded :class:`~arco.mapping.graph.road.RoadGraph`.

    Returns:
        Dict with keys ``route``, ``smooth_path``, ``vehicle``,
        ``controller``, ``loop``, ``goal_xy``.
    """
    start_pos, goal_pos = _find_farthest_pair(graph, list(graph.nodes))

    import numpy as np

    start_xy = np.array(
        [start_pos[0] + _ENDPOINT_OFFSET_M, start_pos[1] + _ENDPOINT_OFFSET_M]
    )
    goal_xy = np.array(
        [goal_pos[0] - _ENDPOINT_OFFSET_M, goal_pos[1] - _ENDPOINT_OFFSET_M]
    )

    router = RouteRouter(graph, activation_radius=_ACTIVATION_RADIUS)
    result = router.plan(start_xy, goal_xy)
    if result is None:
        raise RuntimeError(
            "Route planning failed — check start/goal positions and "
            "activation_radius."
        )

    smooth_path = _build_smooth_path(graph, result.path)
    x0, y0 = smooth_path[0]
    theta0 = _initial_heading(smooth_path)

    vehicle = DubinsVehicle(
        x=x0,
        y=y0,
        heading=theta0,
        max_speed=float(_veh_cfg["max_speed"]),
        min_speed=0.0,
        max_turn_rate=math.radians(float(_veh_cfg["max_turn_rate"])),
        max_acceleration=float(_veh_cfg["max_accel"]),
        max_turn_rate_dot=math.radians(float(_veh_cfg["max_turn_rate_dot"])),
    )
    controller = PurePursuitController(
        lookahead_distance=float(_veh_cfg["lookahead"])
    )
    loop = TrackingLoop(
        vehicle,
        controller,
        cruise_speed=float(_veh_cfg["cruise_speed"]),
        curvature_gain=float(_veh_cfg["curvature_gain"]),
    )

    return {
        "route": result.path,
        "smooth_path": smooth_path,
        "vehicle": vehicle,
        "controller": controller,
        "loop": loop,
        "goal_xy": (float(goal_xy[0]), float(goal_xy[1])),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(fps: int = 30, dt: float = 0.1) -> None:
    """Run the Pygame real-time simulation.

    Args:
        fps: Target frame rate (frames per second).
        dt: Simulation timestep in seconds per frame.
    """
    pygame.init()
    screen_w, screen_h = _resolve_window_size()
    logger.info("Window size: %dx%d", screen_w, screen_h)
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption(TITLE)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)

    # Build road graph (done once; reused across restarts)
    logger.info(
        "Generating %r graph from tools/config/graph.yml",
        _graph_cfg.get("type", "ring"),
    )
    graph = generate_graph(_graph_cfg)
    logger.info(
        "Network generated: %d nodes, %d edges",
        len(graph.nodes),
        len(graph.edges),
    )

    # Compute world-to-screen transform from node positions
    node_positions = [
        (float(graph.position(n)[0]), float(graph.position(n)[1]))
        for n in graph.nodes
    ]
    full_transform = WorldTransform(
        node_positions, (screen_w, screen_h), margin=60
    )

    camera_follow = False
    follow_zoom = _FOLLOW_ZOOM_DEFAULT
    cam_x = 0.0
    cam_y = 0.0
    cam_vx = 0.0
    cam_vy = 0.0

    def _reset_camera_to_vehicle(sim_state) -> None:
        nonlocal cam_x, cam_y, cam_vx, cam_vy
        veh = sim_state["vehicle"]
        cam_x = float(veh.x)
        cam_y = float(veh.y)
        cam_vx = 0.0
        cam_vy = 0.0

    def _update_camera(sim_state, step_dt: float) -> None:
        nonlocal cam_x, cam_y, cam_vx, cam_vy
        veh = sim_state["vehicle"]
        target_x = float(veh.x)
        target_y = float(veh.y)
        wn = _CAMERA_NATURAL_FREQUENCY
        zeta = _CAMERA_DAMPING_RATIO
        ax = (wn * wn) * (target_x - cam_x) - 2.0 * zeta * wn * cam_vx
        ay = (wn * wn) * (target_y - cam_y) - 2.0 * zeta * wn * cam_vy
        cam_vx += ax * step_dt
        cam_vy += ay * step_dt
        cam_x += cam_vx * step_dt
        cam_y += cam_vy * step_dt

    def _current_transform():
        if not camera_follow:
            return full_transform
        return _FollowTransform(
            center_x=cam_x,
            center_y=cam_y,
            screen_size=(screen_w, screen_h),
            scale=full_transform.scale * follow_zoom,
        )

    def restart():
        sim = build_simulation(graph)
        _reset_camera_to_vehicle(sim)
        return sim, [], False, False, 0

    sim, trajectory, finished, paused, step = restart()
    goal_radius = float(_veh_cfg["goal_radius"])

    running = True
    while running:
        # ----------------------------------------------------------------
        # Event handling
        # ----------------------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    sim, trajectory, finished, paused, step = restart()
                elif event.key == pygame.K_c:
                    camera_follow = not camera_follow
                    logger.info(
                        "Camera mode: %s",
                        "follow" if camera_follow else "full",
                    )
                    if camera_follow:
                        _reset_camera_to_vehicle(sim)
                elif event.key in (
                    pygame.K_PLUS,
                    pygame.K_EQUALS,
                    pygame.K_KP_PLUS,
                ):
                    follow_zoom = min(
                        _FOLLOW_ZOOM_MAX,
                        follow_zoom + _FOLLOW_ZOOM_STEP,
                    )
                    logger.info("Follow camera zoom: %.2fx", follow_zoom)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    follow_zoom = max(
                        _FOLLOW_ZOOM_MIN,
                        follow_zoom - _FOLLOW_ZOOM_STEP,
                    )
                    logger.info("Follow camera zoom: %.2fx", follow_zoom)

        # ----------------------------------------------------------------
        # Simulation step
        # ----------------------------------------------------------------
        if not paused and not finished:
            metrics = sim["loop"].step(sim["smooth_path"], dt=dt)
            step += 1
            veh = sim["vehicle"]
            trajectory.append(veh.pose)
            gx, gy = sim["goal_xy"]
            if math.hypot(veh.x - gx, veh.y - gy) < goal_radius:
                finished = True
                logger.info("Goal reached in %d steps.", step)

        if camera_follow:
            _update_camera(sim, dt)

        # ----------------------------------------------------------------
        # Rendering
        # ----------------------------------------------------------------
        screen.fill((28, 28, 35))

        transform = _current_transform()

        draw_road_network(screen, graph, transform, sim["route"])
        draw_smooth_path(screen, sim["smooth_path"], transform)

        if len(trajectory) >= 2:
            draw_trajectory(screen, trajectory, transform)

        veh = sim["vehicle"]
        lookahead = _find_lookahead(
            veh.x, veh.y, sim["smooth_path"], float(_veh_cfg["lookahead"])
        )
        draw_tracking_target(screen, lookahead, transform)
        draw_vehicle(screen, veh.x, veh.y, veh.heading, transform)

        last_metrics = sim["loop"].metrics
        speed = last_metrics["speed"] if last_metrics else 0.0
        cte = last_metrics["cross_track_error"] if last_metrics else 0.0

        draw_hud(screen, font, step, speed, cte, finished, paused)
        draw_legend(screen, font)

        pygame.display.flip()
        clock.tick(fps)

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
        "--dt",
        type=float,
        default=0.1,
        metavar="S",
        help="Simulation timestep in seconds per frame (default: 0.1)",
    )
    args = parser.parse_args()
    main(fps=args.fps, dt=args.dt)
