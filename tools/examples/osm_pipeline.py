#!/usr/bin/env python
"""OSM pipeline: real-world route planning and path tracking on OSM street data.

Demonstrates the complete Phase 1 horse auto-follow pipeline on a real-world
OpenStreetMap street network:

1. **Network import** — :class:`~arco.mapping.importer.OSMImporter` downloads
   the OSM bike network for a bounding box around two Paris landmarks, projects
   it to UTM metres, and converts it to a
   :class:`~arco.mapping.graph.road.RoadGraph`.
2. **Route planning** — :class:`~arco.planning.discrete.RouteRouter` projects
   the start/goal coordinates onto the graph and runs A*.
3. **Path smoothing** — :meth:`~arco.mapping.graph.road.RoadGraph.full_edge_geometry`
   collects the waypoint-dense smooth path from discrete route edges.
4. **Tracking simulation** — :class:`~arco.guidance.tracking.TrackingLoop`
   closes the loop between a :class:`~arco.guidance.vehicle.DubinsVehicle` and
   a :class:`~arco.guidance.pure_pursuit.PurePursuitController`.

The output figure shows:

- OSM street network (projected to UTM metres)
- Planned discrete route (highlighted)
- Smooth path extracted from edge waypoints
- Actual vehicle trajectory
- Cross-track error and speed over simulation time

Usage
-----
Download real OSM data (requires internet access)::

    python tools/examples/osm_pipeline.py
    python tools/examples/osm_pipeline.py --save path/to/output.png

Run in mock mode — uses a procedurally generated network (no network calls,
suitable for CI)::

    python tools/examples/osm_pipeline.py --mock
    python tools/examples/osm_pipeline.py --mock --save path/to/output.png

Configuration is loaded from ``tools/config/osm.yml``.

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
from arco.planning.discrete import RouteRouter
from config import load_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bike-realistic simulation parameters
# ---------------------------------------------------------------------------
CRUISE_SPEED = 4.2  # m/s — 15 km/h, typical urban cycling
LOOKAHEAD = 15.0  # m — scaled to OSM edge lengths
MAX_SPEED = 6.9  # m/s — 25 km/h, French e-bike limit
MAX_TURN_RATE = 0.8  # rad/s — ~45°/s, not a car
MAX_ACCEL = 1.5  # m/s²
MAX_TURN_RATE_DOT = 2.0  # rad/s²

ACTIVATION_RADIUS = (
    None  # No limit — geocoded addresses may be inside buildings or parks,
    # 100-300 m from the nearest road intersection.  The router always
    # projects to the closest node in the graph, regardless of distance.
)

DT = 0.1  # s
MAX_STEPS = 5000
GOAL_RADIUS = 15.0  # m


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
        smooth.extend(tuple(p) for p in pts[:-1])
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
    """Return the lookahead point on *path* at approximately *lookahead* metres.

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


# ---------------------------------------------------------------------------
# Mock network (used when --mock is specified)
# ---------------------------------------------------------------------------


def _build_mock_network():
    """Return a procedurally generated network mimicking the OSM pipeline.

    Used in CI / headless environments where network access is not available.

    Returns:
        Tuple of ``(graph, start_xy, goal_xy)`` in metric coordinates.
    """
    from arco.mapping.generator import RoadNetworkGenerator

    generator = RoadNetworkGenerator(seed=42)
    graph = generator.generate_medieval_network(
        center=(300.0, 300.0),
        num_radials=6,
        ring_radii=[50.0, 110.0, 180.0],
        waypoints_per_edge=2,
        curvature=0.2,
        jitter=0.4,
    )
    # Pick two nodes on opposite sides of the outer ring as start/goal
    cx, cy = 300.0, 300.0
    outer_threshold = 180.0 * 0.70
    outer_nodes = [
        nid
        for nid in graph.nodes
        if math.hypot(graph.position(nid)[0] - cx, graph.position(nid)[1] - cy)
        >= outer_threshold
    ]
    if len(outer_nodes) < 2:
        outer_nodes = list(graph.nodes)

    best_dist = -1.0
    best_pair = (outer_nodes[0], outer_nodes[-1])
    for i in range(len(outer_nodes)):
        for j in range(i + 1, len(outer_nodes)):
            xi, yi = graph.position(outer_nodes[i])
            xj, yj = graph.position(outer_nodes[j])
            d = math.hypot(xi - xj, yi - yj)
            if d > best_dist:
                best_dist = d
                best_pair = (outer_nodes[i], outer_nodes[j])

    start_xy = tuple(float(v) for v in graph.position(best_pair[0]))
    goal_xy = tuple(float(v) for v in graph.position(best_pair[1]))
    return graph, start_xy, goal_xy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(save_path: str | None = None, mock: bool = False) -> None:
    """Run the OSM pipeline and render results.

    Args:
        save_path: File path for saving the figure.  When *None* the figure
            is shown interactively.
        mock: When *True*, use a procedurally generated network instead of
            downloading from OSM.  No network access is required.
    """
    if save_path is not None:
        matplotlib.use("Agg")

    logger.info("=" * 60)
    logger.info("OSM Pipeline \u2014 Horse Auto-Follow System")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load road network
    # ------------------------------------------------------------------
    if mock:
        logger.info("[1] Mock mode \u2014 generating procedural network")
        graph, start_xy, goal_xy = _build_mock_network()
        map_title = "OSM Pipeline \u2014 mock network"
    else:
        cfg = load_config("osm")
        source = cfg.get("source", "address")

        # Lazy import so the script works without osmnx when --mock is used.
        from arco.mapping.importer import OSMImporter

        importer = OSMImporter()

        if source == "address":
            logger.info(
                "[1] Downloading OSM network between '%s' and '%s'",
                cfg["address_start"],
                cfg["address_goal"],
            )
            graph, start_xy, goal_xy = importer.from_addresses(
                address_start=cfg["address_start"],
                address_goal=cfg["address_goal"],
                margin_m=float(cfg.get("margin_m", 500)),
                network_type=cfg.get("network_type", "bike"),
            )
            map_title = (
                f"OSM Pipeline \u2014 {cfg['address_start']} "
                f"\u2192 {cfg['address_goal']}"
            )
        else:
            logger.info(
                "[1] Downloading OSM network between (%.4f, %.4f) "
                "and (%.4f, %.4f)",
                cfg["lat_start"],
                cfg["lon_start"],
                cfg["lat_goal"],
                cfg["lon_goal"],
            )
            graph, start_xy, goal_xy = importer.from_coords(
                lat_start=float(cfg["lat_start"]),
                lon_start=float(cfg["lon_start"]),
                lat_goal=float(cfg["lat_goal"]),
                lon_goal=float(cfg["lon_goal"]),
                margin_m=float(cfg.get("margin_m", 500)),
                network_type=cfg.get("network_type", "bike"),
            )
            map_title = (
                f"OSM Pipeline \u2014 "
                f"({cfg['lat_start']}\u00b0N, {cfg['lon_start']}\u00b0E) "
                f"\u2192 "
                f"({cfg['lat_goal']}\u00b0N, {cfg['lon_goal']}\u00b0E)"
            )

    logger.info(
        "[1] Network ready: %d nodes, %d edges",
        len(graph.nodes),
        len(graph.edges),
    )

    # ------------------------------------------------------------------
    # 2. Route planning
    # ------------------------------------------------------------------
    start_np = np.array(start_xy)
    goal_np = np.array(goal_xy)

    router = RouteRouter(graph, activation_radius=ACTIVATION_RADIUS)
    result = router.plan(start_np, goal_np)

    if result is None:
        logger.error(
            "Route planning failed \u2014 check start/goal positions."
        )
        # Render the network with start/goal markers so the user can inspect
        # the coordinate layout without a full simulation.
        fig, ax_map = plt.subplots(figsize=(10, 8))
        draw_road_network(
            graph, ax=ax_map, title=f"{map_title}\n(route failed)"
        )
        sx, sy = float(start_np[0]), float(start_np[1])
        gx, gy = float(goal_np[0]), float(goal_np[1])
        ax_map.scatter(
            sx, sy, color="limegreen", s=200, zorder=10, label="Start"
        )
        ax_map.scatter(
            gx, gy, color="royalblue", s=200, zorder=10, label="Goal"
        )
        ax_map.legend(loc="upper left", fontsize=8)
        if save_path is not None:
            os.makedirs(
                os.path.dirname(os.path.abspath(save_path)), exist_ok=True
            )
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Partial figure saved \u2192 %s", save_path)
        else:
            plt.show()
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
        max_speed=MAX_SPEED,
        min_speed=0.0,
        max_turn_rate=MAX_TURN_RATE,
        max_acceleration=MAX_ACCEL,
        max_turn_rate_dot=MAX_TURN_RATE_DOT,
    )
    controller = PurePursuitController(lookahead_distance=LOOKAHEAD)
    loop = TrackingLoop(vehicle, controller, cruise_speed=CRUISE_SPEED)

    logger.info(
        "[4] Simulating up to %d steps (dt=%s s) \u2026", MAX_STEPS, DT
    )
    step = 0
    while step < MAX_STEPS:
        loop.step(smooth_path, dt=DT)
        step += 1
        dist_to_goal = math.hypot(vehicle.x - goal_x, vehicle.y - goal_y)
        if dist_to_goal < GOAL_RADIUS:
            break

    history = loop.history
    logger.info(
        "    Completed %d steps \u2014 final distance to goal: %.1f m",
        step,
        math.hypot(vehicle.x - goal_x, vehicle.y - goal_y),
    )

    cross_track = [h["cross_track_error"] for h in history]
    speeds = [h["speed"] for h in history]
    trajectory = [h["pose"] for h in history]

    last_pose = vehicle.pose
    tracking_target = find_lookahead(
        last_pose[0], last_pose[1], smooth_path, LOOKAHEAD
    )

    # ------------------------------------------------------------------
    # 5. Visualisation
    # ------------------------------------------------------------------
    times = [i * DT for i in range(len(history))]

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1], hspace=0.35, wspace=0.3)

    ax_map = fig.add_subplot(gs[:, 0])
    ax_cte = fig.add_subplot(gs[0, 1])
    ax_spd = fig.add_subplot(gs[1, 1])

    draw_road_network(
        graph,
        route=result.path,
        smooth_path=smooth_path,
        trajectory=trajectory,
        tracking_target=tracking_target,
        ax=ax_map,
        title=(
            f"{map_title}\n"
            f"Route: {result.path[0]} \u2192 {result.path[-1]}, "
            f"steps: {step}"
        ),
    )

    ax_cte.plot(times, cross_track, color="crimson", linewidth=1.2)
    ax_cte.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax_cte.set_xlabel("Time (s)")
    ax_cte.set_ylabel("Cross-track error (m)")
    ax_cte.set_title("Cross-track error")
    ax_cte.grid(True, alpha=0.4)

    ax_spd.plot(times, speeds, color="teal", linewidth=1.2)
    ax_spd.axhline(
        CRUISE_SPEED,
        color="gray",
        linewidth=0.8,
        linestyle="--",
        label=f"Cruise {CRUISE_SPEED} m/s",
    )
    ax_spd.set_xlabel("Time (s)")
    ax_spd.set_ylabel("Speed (m/s)")
    ax_spd.set_title("Vehicle speed")
    ax_spd.legend(fontsize=8)
    ax_spd.grid(True, alpha=0.4)

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
        help="Save the figure to PATH instead of opening an interactive window.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help=(
            "Use a procedurally generated network instead of downloading "
            "from OSM. No internet access required (CI-safe)."
        ),
    )
    args = parser.parse_args()
    main(save_path=args.save, mock=args.mock)
