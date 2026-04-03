#!/usr/bin/env python
"""A* pipeline: hand-crafted city road network → route planning → path tracking.

Demonstrates the complete A* horse auto-follow pipeline on the city road
network (``tools/config/city_network.json``):

1. **Road loading** — :func:`~arco.mapping.graph.loader.load_road_graph`
   deserialises the hand-crafted city network descriptor into a
   :class:`~arco.mapping.graph.road.RoadGraph` with 20 intersections and
   40 road segments, each carrying S-curve geometry waypoints.
2. **Route planning** — :class:`~arco.planning.discrete.RouteRouter` projects
   continuous start/goal positions onto graph nodes and runs A*.
3. **Path smoothing** — :meth:`~arco.mapping.graph.road.RoadGraph.full_edge_geometry`
   collects the waypoint-dense smooth path from discrete route edges.
4. **Tracking simulation** — :class:`~arco.guidance.tracking.TrackingLoop`
   closes the loop between a :class:`~arco.guidance.vehicle.DubinsVehicle` and
   a :class:`~arco.guidance.pure_pursuit.PurePursuitController`.

The output figure shows:

- Road graph with curved edge geometry (two-ring city layout)
- Planned discrete route (highlighted)
- Smooth path extracted from edge waypoints
- Actual vehicle trajectory
- Cross-track error, speed, and curvature over simulation time

Usage
-----
Run interactively (opens a Matplotlib window)::

    python tools/examples/astar_pipeline.py

Save to file without opening a window (headless / CI mode)::

    python tools/examples/astar_pipeline.py --save path/to/output.png

Reference
---------
https://alexandrelheinen.github.io/articles/2026-03-06-horse-auto-follow/
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys

# Make the package importable when running the script directly (without install).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
# Expose the tools/viewer and tools/config packages.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from logging_config import configure_logging
from viewer.road import draw_road_network

from arco.guidance.pure_pursuit import PurePursuitController
from arco.guidance.tracking import TrackingLoop
from arco.guidance.vehicle import DubinsVehicle
from arco.mapping.graph.loader import load_road_graph
from arco.planning.discrete import RouteRouter
from config import load_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_sim_cfg = load_config("simulation")
_veh_cfg = load_config("vehicle")["dubins"]

_NETWORK_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "city_network.json"
)


# ---------------------------------------------------------------------------
# Helper: build smooth path from route edge geometry
# ---------------------------------------------------------------------------


def build_smooth_path(graph, route: list[int]) -> list[tuple[float, float]]:
    """Collect the full waypoint-dense path along the discrete route.

    Concatenates :meth:`~arco.mapping.graph.road.RoadGraph.full_edge_geometry`
    for each consecutive pair of nodes in *route*, avoiding endpoint
    duplication.

    Args:
        graph: Road graph with edge geometry metadata.
        route: Ordered sequence of node IDs from start to goal.

    Returns:
        List of ``(x, y)`` waypoints forming the smooth path.
    """
    if len(route) < 2:
        return [tuple(graph.position(route[0]))] if route else []

    smooth: list[tuple[float, float]] = []
    for i in range(len(route) - 1):
        pts = graph.full_edge_geometry(route[i], route[i + 1])
        smooth.extend(pts[:-1])
    smooth.append(tuple(graph.position(route[-1])))
    return smooth


# ---------------------------------------------------------------------------
# Helper: initial heading toward first path segment
# ---------------------------------------------------------------------------


def initial_heading(path: list[tuple[float, float]]) -> float:
    """Return the heading angle pointing from path[0] to path[1].

    Args:
        path: Smooth path as a list of ``(x, y)`` waypoints.

    Returns:
        Heading angle in radians, or 0.0 when the path has fewer than 2 points.
    """
    if len(path) < 2:
        return 0.0
    dx = path[1][0] - path[0][0]
    dy = path[1][1] - path[0][1]
    return math.atan2(dy, dx)


def find_lookahead(
    x: float,
    y: float,
    path: list[tuple[float, float]],
    lookahead: float,
) -> tuple[float, float]:
    """Return the lookahead point on *path* at approximately *lookahead* metres ahead.

    Searches forward along the path from the closest waypoint and returns the
    first point that lies at least *lookahead* metres from ``(x, y)``.
    Falls back to the final path point when no such point is found.

    Args:
        x: Vehicle x position.
        y: Vehicle y position.
        path: Ordered list of ``(x, y)`` waypoints.
        lookahead: Desired lookahead distance (metres).

    Returns:
        ``(x, y)`` coordinates of the lookahead point.
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


def find_farthest_outer_pair(
    graph, outer_nodes: list[int]
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return positions of the two farthest-apart nodes in *outer_nodes*.

    Args:
        graph: Road graph.
        outer_nodes: Candidate node IDs to search.

    Returns:
        Tuple of ``(start_xy, goal_xy)`` for the farthest pair.
    """
    best_dist = -1.0
    best_pair: tuple[int, int] = (outer_nodes[0], outer_nodes[-1])
    for i in range(len(outer_nodes)):
        for j in range(i + 1, len(outer_nodes)):
            xi, yi = graph.position(outer_nodes[i])
            xj, yj = graph.position(outer_nodes[j])
            d = math.hypot(xi - xj, yi - yj)
            if d > best_dist:
                best_dist = d
                best_pair = (outer_nodes[i], outer_nodes[j])
    return (
        tuple(graph.position(best_pair[0])),
        tuple(graph.position(best_pair[1])),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(save_path: str | None = None) -> None:
    """Run the A* pipeline and render results.

    Args:
        save_path: File path for saving the figure.  When *None* the figure
            is shown interactively.
    """
    if save_path is not None:
        matplotlib.use("Agg")

    logger.info("=" * 60)
    logger.info("A* Pipeline \u2014 City Road Network")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load hand-crafted city road network
    # ------------------------------------------------------------------
    graph = load_road_graph(_NETWORK_PATH)
    logger.info(
        "[1] City network loaded: %d nodes, %d edges",
        len(graph.nodes),
        len(graph.edges),
    )

    # ------------------------------------------------------------------
    # 2. Route planning — start/goal at two farthest terminal nodes
    # ------------------------------------------------------------------
    # Terminal nodes are IDs 16-19 (N, E, S, W periphery of the layout).
    outer_node_ids = list(range(16, 20))
    start_pos, goal_pos = find_farthest_outer_pair(graph, outer_node_ids)

    # Small offset so the vehicle starts slightly off a node (tests projection)
    start_xy = np.array([start_pos[0] + 4.0, start_pos[1] + 4.0])
    goal_xy = np.array([goal_pos[0] - 4.0, goal_pos[1] - 4.0])

    router = RouteRouter(
        graph,
        activation_radius=float(_sim_cfg.get("activation_radius", 35.0)),
    )
    result = router.plan(start_xy, goal_xy)

    if result is None:
        logger.error(
            "Route planning failed \u2014 check start/goal positions."
        )
        return

    logger.info(
        "[2] Route planned: %d nodes  (%s \u2192 %s)",
        len(result.path),
        result.path[0],
        result.path[-1],
    )

    # ------------------------------------------------------------------
    # 3. Path smoothing
    # ------------------------------------------------------------------
    smooth_path = build_smooth_path(graph, result.path)
    logger.info("[3] Smooth path: %d waypoints", len(smooth_path))

    # ------------------------------------------------------------------
    # 4. Tracking simulation
    # ------------------------------------------------------------------
    x0, y0 = smooth_path[0]
    theta0 = initial_heading(smooth_path)
    goal_x, goal_y = smooth_path[-1]

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

    logger.info(
        "[4] Simulating up to %d steps (dt=%s s) \u2026",
        int(_sim_cfg["max_steps"]),
        float(_sim_cfg["timestep"]),
    )
    step = 0
    while step < int(_sim_cfg["max_steps"]):
        loop.step(smooth_path, dt=float(_sim_cfg["timestep"]))
        step += 1
        dist_to_goal = math.hypot(vehicle.x - goal_x, vehicle.y - goal_y)
        if dist_to_goal < float(_veh_cfg["goal_radius"]):
            break

    history = loop.history
    logger.info(
        "    Completed %d steps \u2014 final distance to goal: %.1f m",
        step,
        math.hypot(vehicle.x - goal_x, vehicle.y - goal_y),
    )

    cross_track = [h["cross_track_error"] for h in history]
    speeds = [h["speed"] for h in history]
    curvatures = [h["curvature"] for h in history]
    trajectory = [h["pose"] for h in history]

    # Final lookahead target (for display)
    last_pose = vehicle.pose
    tracking_target = find_lookahead(
        last_pose[0],
        last_pose[1],
        smooth_path,
        float(_veh_cfg["lookahead"]),
    )

    # ------------------------------------------------------------------
    # 5. Visualisation
    # ------------------------------------------------------------------
    times = [i * float(_sim_cfg["timestep"]) for i in range(len(history))]

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.6, 1], hspace=0.45, wspace=0.3)

    ax_map = fig.add_subplot(gs[:, 0])
    ax_cte = fig.add_subplot(gs[0, 1])
    ax_spd = fig.add_subplot(gs[1, 1])
    ax_kap = fig.add_subplot(gs[2, 1])

    # Road network + overlays
    draw_road_network(
        graph,
        route=result.path,
        smooth_path=smooth_path,
        trajectory=trajectory,
        tracking_target=tracking_target,
        ax=ax_map,
        title=(
            f"A* Pipeline \u2014 City Road Network\n"
            f"Route: {result.path[0]} \u2192 {result.path[-1]}, "
            f"steps: {step}"
        ),
    )

    # Cross-track error
    ax_cte.plot(times, cross_track, color="crimson", linewidth=1.2)
    ax_cte.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax_cte.set_xlabel("Time (s)")
    ax_cte.set_ylabel("Cross-track error (m)")
    ax_cte.set_title("Cross-track error")
    ax_cte.grid(True, alpha=0.4)

    # Speed
    ax_spd.plot(times, speeds, color="teal", linewidth=1.2)
    ax_spd.axhline(
        float(_veh_cfg["cruise_speed"]),
        color="gray",
        linewidth=0.8,
        linestyle="--",
        label=f"Cruise {float(_veh_cfg['cruise_speed'])} m/s",
    )
    ax_spd.set_xlabel("Time (s)")
    ax_spd.set_ylabel("Speed (m/s)")
    ax_spd.set_title("Vehicle speed")
    ax_spd.legend(fontsize=8)
    ax_spd.grid(True, alpha=0.4)

    # Curvature
    ax_kap.plot(times, curvatures, color="darkorange", linewidth=1.0)
    ax_kap.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax_kap.set_xlabel("Time (s)")
    ax_kap.set_ylabel("Curvature (rad/m)")
    ax_kap.set_title("Path curvature")
    ax_kap.grid(True, alpha=0.4)

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Figure saved \u2192 %s", save_path)
    else:
        plt.show()


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save",
        metavar="PATH",
        default=None,
        help=(
            "Save the figure to PATH instead of opening an interactive "
            "window. Accepts any .png or .pdf path."
        ),
    )
    args = parser.parse_args()
    main(save_path=args.save)
