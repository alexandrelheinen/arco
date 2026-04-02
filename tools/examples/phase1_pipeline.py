#!/usr/bin/env python
"""Phase 1 pipeline: road network generation → route planning → path tracking.

Demonstrates the complete Phase 1 horse auto-follow pipeline:

1. **Road generation** — :class:`~arco.mapping.generator.RoadNetworkGenerator`
   builds a procedural medieval-city-style road network with organic radial
   streets, ring roads, and dead-end alleys.
2. **Route planning** — :class:`~arco.planning.discrete.RouteRouter` projects
   continuous start/goal positions onto graph nodes and runs A*.
3. **Path smoothing** — :meth:`~arco.mapping.graph.road.RoadGraph.full_edge_geometry`
   collects the waypoint-dense smooth path from discrete route edges.
4. **Tracking simulation** — :class:`~arco.guidance.tracking.TrackingLoop`
   closes the loop between a :class:`~arco.guidance.vehicle.DubinsVehicle` and
   a :class:`~arco.guidance.pure_pursuit.PurePursuitController`.

The output figure shows:

- Road graph with curved edge geometry (organic medieval-city layout)
- Planned discrete route (highlighted)
- Smooth path extracted from edge waypoints
- Actual vehicle trajectory
- Cross-track error and speed over simulation time

Usage
-----
Run interactively (opens a Matplotlib window)::

    python tools/examples/phase1_pipeline.py

Save to file without opening a window (headless / CI mode)::

    python tools/examples/phase1_pipeline.py --save path/to/output.png

Reference
---------
https://alexandrelheinen.github.io/articles/2026-03-06-horse-auto-follow/
"""

from __future__ import annotations

import argparse
import math
import os
import sys

# Make the package importable when running the script directly (without install).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
# Expose the tools/viewer package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
import matplotlib.pyplot as plt
from viewer.road import draw_road_network

from arco.guidance.pure_pursuit import PurePursuitController
from arco.guidance.tracking import TrackingLoop
from arco.guidance.vehicle import DubinsVehicle
from arco.mapping.generator import RoadNetworkGenerator
from arco.planning.discrete import RouteRouter

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

SEED = 42

# Medieval city parameters
CITY_CENTER = (200.0, 200.0)
NUM_RADIALS = 7
RING_RADII = [40.0, 90.0, 150.0, 220.0]
WAYPOINTS_PER_EDGE = 1
CURVATURE = 0.25
JITTER = 0.6

ACTIVATION_RADIUS = 30.0

CRUISE_SPEED = 3.0  # m/s
LOOKAHEAD = 12.0  # m
MAX_SPEED = 5.0  # m/s
MAX_TURN_RATE = 1.5  # rad/s
MAX_ACCEL = 4.0  # m/s²
MAX_TURN_RATE_DOT = 4.0  # rad/s²

DT = 0.1  # s
MAX_STEPS = 3000
GOAL_RADIUS = 10.0  # m — stop when vehicle reaches within this distance of goal


# ---------------------------------------------------------------------------
# Helper: build smooth path from route edge geometry
# ---------------------------------------------------------------------------


def build_smooth_path(graph, route: list[int]) -> list[tuple[float, float]]:
    """Collect the full waypoint-dense path along the discrete route.

    Concatenates :meth:`~arco.mapping.graph.road.RoadGraph.full_edge_geometry`
    for each consecutive pair of nodes in *route*, avoiding endpoint duplication.

    Args:
        graph: Road graph with edge geometry metadata.
        route: Ordered sequence of node IDs from start to goal.

    Returns:
        List of ``(x, y)`` waypoints forming the smooth path.
    """
    if len(route) < 2:
        return [graph.position(route[0])] if route else []

    smooth: list[tuple[float, float]] = []
    for i in range(len(route) - 1):
        pts = graph.full_edge_geometry(route[i], route[i + 1])
        if i == 0:
            smooth.extend(pts[:-1])
        else:
            smooth.extend(pts[1:-1])
    smooth.append(graph.position(route[-1]))
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
        range(len(path)), key=lambda i: math.hypot(path[i][0] - x, path[i][1] - y)
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
    return graph.position(best_pair[0]), graph.position(best_pair[1])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(save_path: str | None = None) -> None:
    """Run the Phase 1 pipeline and render results.

    Args:
        save_path: File path for saving the figure.  When *None* the figure
            is shown interactively.
    """
    if save_path is not None:
        matplotlib.use("Agg")

    # ------------------------------------------------------------------
    # 1. Generate road network
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Phase 1 Pipeline — Horse Auto-Follow System")
    print("=" * 60)

    generator = RoadNetworkGenerator(seed=SEED)
    graph = generator.generate_medieval_network(
        center=CITY_CENTER,
        num_radials=NUM_RADIALS,
        ring_radii=RING_RADII,
        waypoints_per_edge=WAYPOINTS_PER_EDGE,
        curvature=CURVATURE,
        jitter=JITTER,
    )
    print(
        f"\n[1] Medieval city network: {len(graph.nodes)} nodes, {len(graph.edges)} edges"
    )

    # ------------------------------------------------------------------
    # 2. Route planning — start/goal at two farthest outer-ring nodes
    # ------------------------------------------------------------------
    # Identify outer nodes by distance: nodes farther than 70% of the outer
    # ring radius are on the outer ring or are dead-end alleys — both are
    # valid route endpoints that span the full city.
    cx, cy = CITY_CENTER
    outer_threshold = RING_RADII[-1] * 0.70
    outer_nodes = [
        nid
        for nid in graph.nodes
        if math.hypot(graph.position(nid)[0] - cx, graph.position(nid)[1] - cy)
        >= outer_threshold
    ]

    start_xy, goal_xy = find_farthest_outer_pair(graph, outer_nodes)
    # Small offset so the vehicle starts slightly off a node (tests projection)
    start_xy = (start_xy[0] + 4.0, start_xy[1] + 4.0)
    goal_xy = (goal_xy[0] - 4.0, goal_xy[1] - 4.0)

    router = RouteRouter(graph, activation_radius=ACTIVATION_RADIUS)
    result = router.plan(*start_xy, *goal_xy)

    if result is None:
        print("ERROR: Route planning failed — check start/goal positions.")
        return

    print(
        f"[2] Route planned: {len(result.path)} nodes"
        f"  ({result.path[0]} → {result.path[-1]})"
    )

    # ------------------------------------------------------------------
    # 3. Path smoothing
    # ------------------------------------------------------------------
    smooth_path = build_smooth_path(graph, result.path)
    print(f"[3] Smooth path: {len(smooth_path)} waypoints")

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

    print(f"[4] Simulating up to {MAX_STEPS} steps (dt={DT} s) …")
    step = 0
    while step < MAX_STEPS:
        loop.step(smooth_path, dt=DT)
        step += 1
        dist_to_goal = math.hypot(vehicle.x - goal_x, vehicle.y - goal_y)
        if dist_to_goal < GOAL_RADIUS:
            break

    history = loop.history
    print(
        f"    Completed {step} steps"
        f" — final distance to goal: {math.hypot(vehicle.x - goal_x, vehicle.y - goal_y):.1f} m"
    )

    cross_track = [h["cross_track_error"] for h in history]
    speeds = [h["speed"] for h in history]
    trajectory = [h["pose"] for h in history]

    # Final lookahead target (for display)
    last_pose = vehicle.pose
    tracking_target = find_lookahead(last_pose[0], last_pose[1], smooth_path, LOOKAHEAD)

    # ------------------------------------------------------------------
    # 5. Visualisation
    # ------------------------------------------------------------------
    times = [i * DT for i in range(len(history))]

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1], hspace=0.35, wspace=0.3)

    ax_map = fig.add_subplot(gs[:, 0])
    ax_cte = fig.add_subplot(gs[0, 1])
    ax_spd = fig.add_subplot(gs[1, 1])

    # Road network + overlays
    draw_road_network(
        graph,
        route=result.path,
        smooth_path=smooth_path,
        trajectory=trajectory,
        tracking_target=tracking_target,
        ax=ax_map,
        title=(
            f"Phase 1 Pipeline — Medieval city network\n"
            f"Route: {result.path[0]} → {result.path[-1]}, "
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
        print(f"\nFigure saved → {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save",
        metavar="PATH",
        default=None,
        help="Save the figure to PATH instead of opening an interactive window. "
        "Accepts any .png or .pdf path.",
    )
    args = parser.parse_args()
    main(save_path=args.save)
