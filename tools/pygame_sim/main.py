#!/usr/bin/env python
"""Real-time Pygame frontend for the ARCO A* horse auto-follow pipeline.

Loads the hand-crafted city road network, plans an A* route between two
terminal nodes, and animates the full planning + tracking loop in real time
using Pygame.

Keyboard controls
-----------------
SPACE       Pause / resume simulation
R           Restart from the beginning
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

    cd tools/pygame_sim
    pip install -r requirements.txt     # installs pygame
    python main.py

    # Or from the repo root (after installing arco):
    python tools/pygame_sim/main.py

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
from arco.mapping.graph.loader import load_road_graph
from arco.planning.discrete import RouteRouter
from config import load_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------
_NETWORK_PATH = os.path.join(_HERE, "..", "config", "city_network.json")
_veh_cfg = load_config("vehicle")["dubins"]

SCREEN_W = 1280
SCREEN_H = 800
TITLE = "ARCO — City Road Network — A* Path Tracking (SPACE=pause, R=restart)"

# Small metric offset applied to start/goal so the vehicle begins slightly
# off a graph node, exercising the projection logic in RouteRouter.
_ENDPOINT_OFFSET_M = 4.0

# Terminal nodes (IDs 16-19: N, E, S, W) are the farthest-apart entry points.
_OUTER_NODE_IDS = list(range(16, 20))
_ACTIVATION_RADIUS = 35.0


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


def build_simulation(graph):
    """Build and return all simulation objects for one run.

    Args:
        graph: Pre-loaded :class:`~arco.mapping.graph.road.RoadGraph`.

    Returns:
        Dict with keys ``route``, ``smooth_path``, ``vehicle``,
        ``controller``, ``loop``, ``goal_xy``.
    """
    start_pos, goal_pos = _find_farthest_pair(graph, _OUTER_NODE_IDS)

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
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption(TITLE)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)

    # Load road graph (done once; reused across restarts)
    logger.info("Loading city road network from %s", _NETWORK_PATH)
    graph = load_road_graph(_NETWORK_PATH)
    logger.info(
        "Network loaded: %d nodes, %d edges",
        len(graph.nodes),
        len(graph.edges),
    )

    # Compute world-to-screen transform from node positions
    node_positions = [
        (float(graph.position(n)[0]), float(graph.position(n)[1]))
        for n in graph.nodes
    ]
    transform = WorldTransform(node_positions, (SCREEN_W, SCREEN_H), margin=60)

    def restart():
        sim = build_simulation(graph)
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

        # ----------------------------------------------------------------
        # Rendering
        # ----------------------------------------------------------------
        screen.fill((28, 28, 35))

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
