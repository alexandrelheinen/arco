"""Solver runs behind the web plates.

Every curve comes from a shipped planner or controller. Sample budgets
stay small because the drawing then throws away all but the longest
tree edges: the picture does not get clearer past a few hundred
samples, and a release job should not spend a minute in RRT*.

Results are pickled under ``tools/output/web_cache`` keyed by the
budgets, so a second render in the same checkout does not re-solve.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from arco.control import PurePursuitController, TrackingLoop
from arco.guidance import DubinsVehicle
from arco.planning import (
    AStarPlanner,
    RRTPlanner,
    SSTPlanner,
    TrajectoryOptimizer,
    TrajectoryPruner,
)

from .budget import Parents
from .scene import Basin, Flood

CACHE_DIR = (
    Path(__file__).resolve().parents[3] / "tools" / "output" / "web_cache"
)

RRT_SAMPLES = 320
SST_SAMPLES = 320
RRT_SEED = 7
SST_SEED = 19
STEP = 6.0


@dataclass
class TreeRun:
    """One sampling-planner run.

    Attributes:
        nodes: ``(N, 2)`` node positions.
        parents: Parent index per node, ``None`` at the root.
        path: Solution polyline ``(M, 2)``.
        label: Planner name, for logs. It is not drawn.
    """

    nodes: np.ndarray
    parents: Parents
    path: np.ndarray
    label: str


@dataclass
class BasinRuns:
    """Everything the basin plates draw, from one pair of solves.

    Attributes:
        rrt: RRT* tree and path.
        sst: SST tree and path. Only the path is drawn.
        curve: Smoothed polyline shared by ``arc`` and ``ribbon``.
        speed: Speed encoded by the ribbon, world units per second.
            Taken from the optimizer when that speed actually varies,
            otherwise the Dubins turn-rate limit along *curve*.
        reference: Reference polyline the tracker followed.
        executed: Executed ``(x, y)`` polyline.
    """

    rrt: TreeRun
    sst: TreeRun
    curve: np.ndarray
    speed: np.ndarray
    reference: np.ndarray
    executed: np.ndarray


@dataclass
class FloodRun:
    """A* expansion and the path it returned.

    Attributes:
        path: Solution cells ``(M, 2)`` as ``(i, j)``.
        expanded: Expanded cells in order, ``(K, 2)`` as ``(i, j)``.
    """

    path: np.ndarray
    expanded: np.ndarray


def load_basin(
    basin: Basin,
    rrt_samples: int = RRT_SAMPLES,
    sst_samples: int = SST_SAMPLES,
    cache_dir: Path = CACHE_DIR,
) -> BasinRuns:
    """Return the basin runs, from the cache when the budgets match.

    Args:
        basin: Scene to plan in.
        rrt_samples: RRT* sample budget.
        sst_samples: SST sample budget.
        cache_dir: Directory of pickled runs.

    Returns:
        The cached or freshly computed runs.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_dir / (
        f"basin_s{RRT_SEED}_{SST_SEED}_n{rrt_samples}_{sst_samples}.pkl"
    )
    if key.exists():
        with key.open("rb") as handle:
            return pickle.load(handle)
    runs = _compute_basin(basin, rrt_samples, sst_samples)
    with key.open("wb") as handle:
        pickle.dump(runs, handle)
    return runs


def load_flood(flood: Flood, cache_dir: Path = CACHE_DIR) -> FloodRun:
    """Return the A* run, from the cache when present.

    Args:
        flood: Grid to search.
        cache_dir: Directory of pickled runs.

    Returns:
        The cached or freshly computed run.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_dir / "flood.pkl"
    if key.exists():
        with key.open("rb") as handle:
            return pickle.load(handle)
    run = _compute_flood(flood)
    with key.open("wb") as handle:
        pickle.dump(run, handle)
    return run


def _compute_basin(
    basin: Basin, rrt_samples: int, sst_samples: int
) -> BasinRuns:
    """Solve RRT*, SST, the smoother and the tracker.

    Args:
        basin: Scene to plan in.
        rrt_samples: RRT* sample budget.
        sst_samples: SST sample budget.

    Returns:
        Runs ready to draw.
    """
    rrt = _tree(
        RRTPlanner(
            basin.occupancy,
            basin.bounds,
            max_sample_count=rrt_samples,
            step_size=STEP,
            goal_tolerance=5.0,
            goal_bias=0.12,
            early_stop=False,
            seed=RRT_SEED,
        ),
        basin,
        "RRT*",
    )
    sst = _tree(
        SSTPlanner(
            basin.occupancy,
            basin.bounds,
            max_sample_count=sst_samples,
            step_size=STEP,
            goal_tolerance=5.0,
            witness_radius=0.45,
            goal_bias=0.12,
            early_stop=False,
            seed=SST_SEED,
        ),
        basin,
        "SST",
    )
    curve, speed = _smooth(basin, rrt.path)
    reference, executed = _track(curve)
    return BasinRuns(
        rrt=rrt,
        sst=sst,
        curve=curve,
        speed=speed,
        reference=reference,
        executed=executed,
    )


def _tree(planner, basin: Basin, label: str) -> TreeRun:
    """Grow one sampling planner and require a path.

    Args:
        planner: Configured ``RRTPlanner`` or ``SSTPlanner``.
        basin: Scene, for the start and the goal.
        label: Name used in the error and the log.

    Returns:
        The completed run.

    Raises:
        RuntimeError: If the planner returns no path.
    """
    nodes, parents, path = planner.get_tree(basin.start, basin.goal)
    if path is None:
        raise RuntimeError(f"{label} found no path in the web basin.")
    return TreeRun(
        nodes=np.asarray(nodes, dtype=float),
        parents=parents,
        path=np.asarray(path, dtype=float),
        label=label,
    )


def _smooth(basin: Basin, path: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Prune and optimize *path*, then resample it with a speed.

    Args:
        basin: Scene the pruner and the optimizer check collisions in.
        path: Raw planner polyline ``(N, 2)``.

    Returns:
        ``(curve, speed)`` with matching lengths. *speed* is world units
        per second from the optimizer segment times.
    """
    pruner = TrajectoryPruner(
        basin.occupancy, step_size=np.array([STEP, STEP])
    )
    pruned = np.asarray(
        pruner.prune([np.asarray(point, dtype=float) for point in path]),
        dtype=float,
    )
    seeded = _densify(pruned, max_step=10.0)
    optimizer = TrajectoryOptimizer(
        basin.occupancy,
        cruise_speed=8.0,
        max_speed=14.0,
        min_speed=2.5,
        weight_time=8.0,
        weight_deviation=1.0,
        weight_collision=4.0,
    )
    result = optimizer.optimize([row for row in seeded])
    states = np.asarray(result.states, dtype=float)
    durations = np.asarray(result.durations, dtype=float)
    curve, segment_speed = _resample_with_speed(
        states, durations, max_step=2.0
    )
    return curve, _ribbon_speed(curve, segment_speed)


def _track(curve: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Drive a Dubins vehicle that starts offset from *curve*.

    The offset plus a modest turn-rate limit is what makes the executed
    line peel away and rejoin. A perfect overlap would leave the plate
    with one stroke.

    Args:
        curve: Reference polyline ``(N, 2)``.

    Returns:
        ``(reference, executed)`` polylines.
    """
    reference = np.asarray(curve, dtype=float)
    span = min(3, len(reference) - 1)
    heading = float(
        np.arctan2(
            reference[span, 1] - reference[0, 1],
            reference[span, 0] - reference[0, 0],
        )
    )
    normal = np.array([-np.sin(heading), np.cos(heading)])
    origin = reference[0] + 7.0 * normal
    vehicle = DubinsVehicle(
        x=float(origin[0]),
        y=float(origin[1]),
        heading=heading,
        max_speed=12.0,
        min_speed=0.0,
        max_turn_rate=0.45,
        max_acceleration=3.5,
        max_turn_rate_dot=1.0,
    )
    controller = PurePursuitController(lookahead_distance=16.0)
    loop = TrackingLoop(vehicle, controller, cruise_speed=7.0)
    waypoints = [(float(x), float(y)) for x, y in reference]
    goal = reference[-1]
    poses: List[np.ndarray] = []
    dt = 0.08
    for _ in range(1600):
        step = loop.step(waypoints, dt)
        poses.append(np.asarray(step["pose"][:2], dtype=float))
        if np.linalg.norm(poses[-1] - goal) < 5.0:
            break
    return reference, np.asarray(poses, dtype=float)


def _compute_flood(flood: Flood) -> FloodRun:
    """Run A* with diagnostics.

    Args:
        flood: Grid to search.

    Returns:
        Path and expansion order as cell indices.

    Raises:
        RuntimeError: If A* finds no path.
    """
    path, expanded, _ = AStarPlanner(
        flood.grid, simplify_path=False
    ).plan_with_diagnostics(flood.start, flood.goal)
    if path is None:
        raise RuntimeError("A* found no path in the web flood grid.")
    return FloodRun(
        path=np.asarray(path, dtype=int),
        expanded=np.asarray(expanded, dtype=int),
    )


def _ribbon_speed(curve: np.ndarray, segment_speed: np.ndarray) -> np.ndarray:
    """Return the speed the ribbon thickness encodes.

    The optimizer's own segment speeds are used when they actually vary.
    On this scene they sit near cruise, because the dynamics term only
    penalizes leaving the speed band. The thickness then follows the
    speed a Dubins vehicle can hold on the curve: cruise on the straight
    sections, slower where the curvature would exceed ``max_turn_rate``.

    Args:
        curve: Smoothed polyline ``(N, 2)``.
        segment_speed: Optimizer speed at each point of *curve*.

    Returns:
        Speed samples, world units per second, length ``N``.
    """
    low = float(np.min(segment_speed))
    high = float(np.max(segment_speed))
    if low > 1e-6 and high / low >= 1.35:
        return segment_speed
    return _curvature_speed(
        curve, cruise=8.0, floor=2.5, max_turn_rate=0.55, window=8.0
    )


def _curvature_speed(
    curve: np.ndarray,
    cruise: float,
    floor: float,
    max_turn_rate: float,
    window: float,
) -> np.ndarray:
    """Return the fastest speed that respects a turn-rate limit.

    Curvature is dilated by *window* world units along the curve, so the
    slowdown covers the bend instead of a single sample.

    Args:
        curve: Polyline ``(N, 2)``.
        cruise: Speed used where the curve is straight.
        floor: Slowest speed the ribbon will show.
        max_turn_rate: Dubins turn-rate limit, radians per second.
        window: Arc-length radius of the curvature dilation.

    Returns:
        Speed at each point of *curve*.
    """
    curvature = np.zeros(len(curve))
    for index in range(1, len(curve) - 1):
        before = curve[index] - curve[index - 1]
        after = curve[index + 1] - curve[index]
        length_before = float(np.linalg.norm(before))
        length_after = float(np.linalg.norm(after))
        if length_before < 1e-6 or length_after < 1e-6:
            continue
        cross = float(before[0] * after[1] - before[1] * after[0])
        mean = 0.5 * (length_before + length_after)
        curvature[index] = abs(cross) / (length_before * length_after * mean)
    if len(curve) > 2:
        curvature[0] = curvature[1]
        curvature[-1] = curvature[-2]
    if len(curve) > 1:
        segment = np.linalg.norm(np.diff(curve, axis=0), axis=1)
        distance = np.concatenate([[0.0], np.cumsum(segment)])
        dilated = np.zeros_like(curvature)
        for index, point in enumerate(distance):
            dilated[index] = float(
                curvature[np.abs(distance - point) <= window].max()
            )
        curvature = dilated
    limited = max_turn_rate / np.maximum(curvature, 1e-3)
    return np.clip(np.minimum(cruise, limited), floor, cruise)


def _densify(points: np.ndarray, max_step: float) -> np.ndarray:
    """Insert points so no segment is longer than *max_step*.

    Args:
        points: ``(N, 2)`` polyline.
        max_step: Maximum segment length in world units.

    Returns:
        The densified polyline.
    """
    out: List[np.ndarray] = [np.asarray(points[0], dtype=float)]
    for start, end in zip(points[:-1], points[1:]):
        length = float(np.linalg.norm(end - start))
        pieces = max(int(np.ceil(length / max_step)), 1)
        for step in range(1, pieces + 1):
            out.append(start + (end - start) * (step / pieces))
    return np.asarray(out, dtype=float)


def _resample_with_speed(
    states: np.ndarray,
    durations: np.ndarray,
    max_step: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample an optimized polyline and carry the segment speed along.

    Args:
        states: Optimized waypoints ``(N, 2)``.
        durations: Per-segment times, length ``N - 1``.
        max_step: Maximum spacing of the resampled curve.

    Returns:
        ``(curve, speed)`` of equal length.
    """
    speeds = []
    for start, end, duration in zip(states[:-1], states[1:], durations):
        length = float(np.linalg.norm(end - start))
        speeds.append(length / max(float(duration), 1e-6))
    curve: List[np.ndarray] = [states[0]]
    sampled: List[float] = [speeds[0] if speeds else 0.0]
    for index, (start, end) in enumerate(zip(states[:-1], states[1:])):
        length = float(np.linalg.norm(end - start))
        pieces = max(int(np.ceil(length / max_step)), 1)
        for step in range(1, pieces + 1):
            curve.append(start + (end - start) * (step / pieces))
            sampled.append(speeds[index])
    return np.asarray(curve, dtype=float), np.asarray(sampled, dtype=float)
