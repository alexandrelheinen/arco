"""Integration tests for the Phase 1 horse auto-follow pipeline.

These tests validate that the full pipeline runs end-to-end and that the
tracking controller converges with acceptable metrics within a finite time
budget.  They do *not* pin absolute numeric outputs — only qualitative
properties such as convergence, stability, and progress are checked.

The road network is loaded from ``tools/config/city_network.json``, the
hand-crafted triangular-mesh city graph that replaces the procedural generator.
"""

from __future__ import annotations

import math
import os
import time

import numpy as np
import pytest

from arco.guidance.control.pure_pursuit import PurePursuitController
from arco.guidance.control.tracking import TrackingLoop
from arco.guidance.vehicle import DubinsVehicle
from arco.mapping.graph.loader import load_road_graph
from arco.planning.discrete import RouteRouter

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------


def _resolve_network_path() -> str:
    """Return the existing city-network descriptor path.

    Supports both the legacy ``city_network.json`` and the newer
    ``city.json`` filenames.
    """
    config_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "src",
        "arco",
        "tools",
        "config",
    )
    for filename in ("city_network.json", "city.json"):
        path = os.path.join(config_dir, filename)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "City network descriptor not found in tools/config/. "
        "Expected city_network.json or city.json."
    )


_NETWORK_PATH = _resolve_network_path()

# Terminal N (id=57) is at (365, 1070); terminal S (id=59) is at (365, -70).
# Small offsets so the vehicle starts slightly off a graph node.
START_XY = (369.0, 1066.0)
GOAL_XY = (361.0, -74.0)

ACTIVATION_RADIUS = 30.0
CRUISE_SPEED = 5.0
LOOKAHEAD = 15.0
DT = 0.1
MAX_STEPS = 5000


def _build_smooth_path(graph, route: list[int]) -> list[tuple[float, float]]:
    """Collect dense smooth path from route edge geometry."""
    if len(route) < 2:
        return [tuple(graph.position(route[0]))] if route else []
    smooth: list[tuple[float, float]] = []
    for i in range(len(route) - 1):
        pts = graph.full_edge_geometry(route[i], route[i + 1])
        smooth.extend(pts[:-1])
    smooth.append(tuple(graph.position(route[-1])))
    return smooth


@pytest.fixture(scope="module")
def pipeline():
    """Build the complete pipeline once and return all artifacts."""
    graph = load_road_graph(_NETWORK_PATH)
    router = RouteRouter(graph, activation_radius=ACTIVATION_RADIUS)
    result = router.plan(np.array(START_XY), np.array(GOAL_XY))
    return {
        "graph": graph,
        "result": result,
    }


@pytest.fixture(scope="module")
def simulation(pipeline):
    """Run the tracking simulation once; expose history."""
    graph = pipeline["graph"]
    result = pipeline["result"]
    assert result is not None, "Route planning must succeed before simulation"

    smooth_path = _build_smooth_path(graph, result.path)
    x0, y0 = smooth_path[0]
    if len(smooth_path) >= 2:
        dx = smooth_path[1][0] - smooth_path[0][0]
        dy = smooth_path[1][1] - smooth_path[0][1]
        theta0 = math.atan2(dy, dx)
    else:
        theta0 = 0.0

    vehicle = DubinsVehicle(
        x=x0,
        y=y0,
        heading=theta0,
        max_speed=10.0,
        min_speed=0.0,
        max_turn_rate=1.5,
        max_acceleration=4.0,
        max_turn_rate_dot=4.0,
    )
    controller = PurePursuitController(lookahead_distance=LOOKAHEAD)
    loop = TrackingLoop(vehicle, controller, cruise_speed=CRUISE_SPEED)

    goal_x, goal_y = smooth_path[-1]
    step = 0
    while step < MAX_STEPS:
        loop.step(smooth_path, dt=DT)
        step += 1
        if math.hypot(vehicle.x - goal_x, vehicle.y - goal_y) < LOOKAHEAD:
            break

    return {
        "loop": loop,
        "vehicle": vehicle,
        "smooth_path": smooth_path,
        "steps": step,
        "goal": (goal_x, goal_y),
    }


# ---------------------------------------------------------------------------
# Stage 1 — Road network loading
# ---------------------------------------------------------------------------


def test_graph_has_nodes_and_edges(pipeline) -> None:
    """Loaded graph must have the expected number of nodes and edges."""
    graph = pipeline["graph"]
    assert len(graph.nodes) == 61
    assert len(graph.edges) == 110


def test_graph_edges_have_waypoints(pipeline) -> None:
    """Every edge in the road graph must carry at least one waypoint."""
    graph = pipeline["graph"]
    for a, b, _ in graph.edges:
        pts = graph.edge_geometry(a, b)
        assert len(pts) >= 1, f"Edge ({a},{b}) has no waypoints"


# ---------------------------------------------------------------------------
# Stage 2 — Route planning
# ---------------------------------------------------------------------------


def test_route_planning_succeeds(pipeline) -> None:
    """RouteRouter must find a route between the specified start and goal."""
    assert pipeline["result"] is not None


def test_route_path_non_trivial(pipeline) -> None:
    """Planned path must visit more than one node."""
    result = pipeline["result"]
    assert result is not None
    assert len(result.path) >= 2


def test_route_connects_start_to_goal(pipeline) -> None:
    """Path must start near START_XY and end near GOAL_XY."""
    graph = pipeline["graph"]
    result = pipeline["result"]
    assert result is not None
    start_node_pos = graph.position(result.path[0])
    goal_node_pos = graph.position(result.path[-1])
    assert (
        math.hypot(
            start_node_pos[0] - START_XY[0],
            start_node_pos[1] - START_XY[1],
        )
        <= ACTIVATION_RADIUS
    )
    assert (
        math.hypot(
            goal_node_pos[0] - GOAL_XY[0],
            goal_node_pos[1] - GOAL_XY[1],
        )
        <= ACTIVATION_RADIUS
    )


# ---------------------------------------------------------------------------
# Stage 3 — Path smoothing
# ---------------------------------------------------------------------------


def test_smooth_path_has_enough_waypoints(pipeline) -> None:
    """Smooth path must contain at least as many points as the route has nodes."""
    graph = pipeline["graph"]
    result = pipeline["result"]
    assert result is not None
    smooth = _build_smooth_path(graph, result.path)
    min_expected = len(result.path)
    assert len(smooth) >= min_expected


def test_smooth_path_starts_and_ends_at_route_nodes(pipeline) -> None:
    """Smooth path endpoints must coincide with the first/last route nodes."""
    graph = pipeline["graph"]
    result = pipeline["result"]
    assert result is not None
    smooth = _build_smooth_path(graph, result.path)
    assert np.array_equal(smooth[0], graph.position(result.path[0]))
    assert np.array_equal(smooth[-1], graph.position(result.path[-1]))


# ---------------------------------------------------------------------------
# Stage 4 — Tracking simulation
# ---------------------------------------------------------------------------


def test_simulation_runs_in_finite_steps(simulation) -> None:
    """Simulation must complete within MAX_STEPS."""
    assert simulation["steps"] <= MAX_STEPS


def test_simulation_runs_in_finite_wall_time() -> None:
    """Full pipeline including simulation must complete in under 30 seconds."""
    start = time.monotonic()

    graph = load_road_graph(_NETWORK_PATH)
    router = RouteRouter(graph, activation_radius=ACTIVATION_RADIUS)
    result = router.plan(np.array(START_XY), np.array(GOAL_XY))
    assert result is not None

    smooth = _build_smooth_path(graph, result.path)
    x0, y0 = smooth[0]
    theta0 = math.atan2(
        smooth[1][1] - smooth[0][1], smooth[1][0] - smooth[0][0]
    )
    vehicle = DubinsVehicle(
        x=x0,
        y=y0,
        heading=theta0,
        max_speed=10.0,
        min_speed=0.0,
        max_turn_rate=1.5,
        max_acceleration=4.0,
        max_turn_rate_dot=4.0,
    )
    loop = TrackingLoop(
        vehicle,
        PurePursuitController(LOOKAHEAD),
        cruise_speed=CRUISE_SPEED,
    )
    loop.run(smooth, steps=1000, dt=DT)

    elapsed = time.monotonic() - start
    assert elapsed < 30.0, f"Pipeline took {elapsed:.1f} s (limit 30 s)"


def test_vehicle_speed_bounded_throughout(simulation) -> None:
    """Vehicle speed must remain within [0, max_speed] for all steps."""
    for entry in simulation["loop"].history:
        assert 0.0 - 1e-9 <= entry["speed"] <= 10.0 + 1e-9


def test_vehicle_turn_rate_bounded_throughout(simulation) -> None:
    """Vehicle turn rate must remain within ±max_turn_rate for all steps."""
    for entry in simulation["loop"].history:
        assert abs(entry["turn_rate"]) <= 1.5 + 1e-9


def test_cross_track_errors_are_finite(simulation) -> None:
    """Cross-track error must be a finite float at every simulation step."""
    for entry in simulation["loop"].history:
        assert math.isfinite(entry["cross_track_error"])


def test_vehicle_makes_forward_progress(simulation) -> None:
    """Vehicle must cover at least half the straight-line start-to-goal distance."""
    smooth = simulation["smooth_path"]
    history = simulation["loop"].history
    start_pos = smooth[0]
    goal_pos = smooth[-1]
    straight_dist = math.hypot(
        goal_pos[0] - start_pos[0], goal_pos[1] - start_pos[1]
    )
    final_pose = history[-1]["pose"]
    dist_covered = math.hypot(
        final_pose[0] - start_pos[0], final_pose[1] - start_pos[1]
    )
    assert dist_covered > straight_dist * 0.5, (
        f"Vehicle only traveled {dist_covered:.1f} m of "
        f"{straight_dist:.1f} m straight-line distance"
    )


def test_cross_track_error_reduces_over_time(simulation) -> None:
    """Cross-track error in the last quarter must not exceed 3× the first quarter average, with a minimum tolerance of 30 m."""
    history = simulation["loop"].history
    n = len(history)
    if n < 40:
        pytest.skip("Simulation too short to assess convergence")

    quarter = n // 4
    early_errors = [abs(h["cross_track_error"]) for h in history[:quarter]]
    late_errors = [abs(h["cross_track_error"]) for h in history[-quarter:]]
    avg_early = sum(early_errors) / len(early_errors)
    avg_late = sum(late_errors) / len(late_errors)
    assert avg_late <= max(avg_early * 3.0, 30.0), (
        f"Cross-track error did not reduce: early avg={avg_early:.2f} m, "
        f"late avg={avg_late:.2f} m"
    )


def test_vehicle_reaches_goal_vicinity(simulation) -> None:
    """Vehicle must reach within 2× LOOKAHEAD of the goal waypoint."""
    final_pose = simulation["loop"].history[-1]["pose"]
    goal = simulation["goal"]
    dist = math.hypot(final_pose[0] - goal[0], final_pose[1] - goal[1])
    assert dist < LOOKAHEAD * 2, (
        f"Vehicle ended {dist:.1f} m from goal "
        f"(tolerance {LOOKAHEAD * 2:.1f} m)"
    )
