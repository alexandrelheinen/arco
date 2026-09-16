"""Solution: every curve the gallery draws, produced by the real solvers.

Nothing here fabricates geometry.  Trees come from
:meth:`~arco.planning.RRTPlanner.get_tree`, the wavefront from
:meth:`~arco.planning.AStarPlanner.plan_with_diagnostics`, the smooth
trajectory from :class:`~arco.planning.TrajectoryOptimizer`, and the
executed motion from :class:`~arco.control.TrackingLoop` driving a
:class:`~arco.guidance.DubinsVehicle`.

Runs are slow enough (a dense RRT* is tens of seconds) that results are
pickled under ``tools/output/gallery_cache`` and keyed by their own
parameters, so re-rendering a plate after a styling change is instant.

One private import is deliberate: ``pure_pursuit._find_lookahead`` is how
the tracking plate draws the lookahead points the controller actually
used.  Re-deriving them here would let the picture drift away from the
control law it claims to show.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from arco.control import (
    ArtificialPotentialField,
    PurePursuitController,
    TrackingLoop,
)
from arco.control.pure_pursuit import _find_lookahead
from arco.guidance import DubinsVehicle
from arco.planning import (
    AStarPlanner,
    RRTPlanner,
    SSTPlanner,
    TrajectoryOptimizer,
    TrajectoryPruner,
)

from .world import World

CACHE_DIR = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "output"
    / ("gallery_cache")
)


@dataclass
class TreeRun:
    """One sampling-planner run kept whole, tree included.

    Attributes:
        nodes: ``(N, 2)`` node positions in world units.
        parents: Node index to parent index (``None`` at the root).
        path: Solution polyline as ``(M, 2)``, or ``None`` if unsolved.
        sample_count: Sample budget the planner was given.
        seconds: Wall-clock time of the run.
        label: Display name of the planner.
    """

    nodes: np.ndarray
    parents: Dict[int, Optional[int]]
    path: Optional[np.ndarray]
    sample_count: int
    seconds: float
    label: str

    @property
    def length(self) -> float:
        """Return the solution path length in world units (0 if unsolved)."""
        return _polyline_length(self.path)


@dataclass
class GridRun:
    """One A* run with the exploration it needed to get there.

    Attributes:
        path: Solution polyline as ``(M, 2)`` in world units.
        expanded: ``(K, 2)`` expanded cell centres in expansion order.
        seconds: Wall-clock time of the run.
        cell_count: Number of free cells in the grid.
    """

    path: np.ndarray
    expanded: np.ndarray
    seconds: float
    cell_count: int

    @property
    def length(self) -> float:
        """Return the solution path length in world units."""
        return _polyline_length(self.path)


@dataclass
class Refinement:
    """The planning-to-guidance chain applied to one raw path.

    Attributes:
        raw: Planner output polyline ``(N, 2)``.
        pruned: Shortcut-pruned polyline ``(M, 2)``.
        states: Optimized waypoints ``(M, 2)``.
        durations: Per-segment traversal times (seconds).
        dense: Resampled optimized polyline ``(P, 2)`` for stroking.
        dense_speed: Speed at each point of *dense* (world units / s).
        cost: Composite optimizer cost at the solution.
        seconds: Wall-clock time of prune plus optimize.
    """

    raw: np.ndarray
    pruned: np.ndarray
    states: np.ndarray
    durations: np.ndarray
    dense: np.ndarray
    dense_speed: np.ndarray
    cost: float
    seconds: float


@dataclass
class TrackingRun:
    """A closed-loop run of a Dubins vehicle over a reference path.

    Attributes:
        reference: Reference polyline ``(N, 2)``.
        poses: Executed poses ``(T, 3)`` as ``(x, y, heading)``.
        speed: Executed speed per step ``(T,)``.
        cross_track: Signed lateral error per step ``(T,)``.
        repulsion: APF turn-rate correction per step ``(T,)``.
        carrots: Pure-pursuit lookahead points ``(T, 2)``.
        dt: Integration step in seconds.
    """

    reference: np.ndarray
    poses: np.ndarray
    speed: np.ndarray
    cross_track: np.ndarray
    repulsion: np.ndarray
    carrots: np.ndarray
    dt: float


@dataclass
class Reachability:
    """The set of motions one vehicle model admits from a single pose.

    Every curve is a forward integration of
    :meth:`~arco.guidance.DubinsVehicle.step` under a turn-rate ramp, so
    the fan shows the model's real acceleration and turn-rate-dot limits
    rather than an idealised arc bundle.

    Attributes:
        curves: Rollouts, each ``(T, 2)``.
        free: Per-rollout flag, ``True`` when the whole rollout stays
            clear of the occupancy map.
        turn_rate: Terminal turn-rate command of each rollout (rad/s).
        origin: Common start pose ``(x, y, heading)``.
        horizon: Rollout duration in seconds.
    """

    curves: List[np.ndarray] = field(default_factory=list)
    free: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    turn_rate: np.ndarray = field(default_factory=lambda: np.zeros(0))
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    horizon: float = 0.0


class Solution:
    """Lazily computed, disk-cached solver output for one :class:`World`.

    Args:
        world: Scene every run is planned in.
        cache_dir: Directory holding pickled run results.
    """

    def __init__(self, world: World, cache_dir: Path = CACHE_DIR) -> None:
        """Create the accessor and ensure the cache directory exists."""
        self.world = world
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def rrt(self, sample_count: int = 7000, seed: int = 7) -> TreeRun:
        """Return an RRT* run, growing the tree to *sample_count* samples.

        Args:
            sample_count: Sample budget; the tree is grown to exhaustion
                (``early_stop=False``) so the plate shows the full
                asymptotic structure rather than a first solution.
            seed: Planner RNG seed.

        Returns:
            The cached or freshly computed :class:`TreeRun`.
        """
        return self._cached(
            f"rrt_{sample_count}_{seed}",
            lambda: self._tree_run(
                RRTPlanner(
                    self.world.occupancy,
                    self.world.bounds,
                    max_sample_count=sample_count,
                    step_size=3.4,
                    goal_tolerance=3.0,
                    goal_bias=0.04,
                    early_stop=False,
                    seed=seed,
                ),
                sample_count,
                "RRT*",
            ),
        )

    def sst(self, sample_count: int = 7000, seed: int = 5) -> TreeRun:
        """Return an SST run over the same scene.

        Args:
            sample_count: Propagation budget.
            seed: Planner RNG seed.

        Returns:
            The cached or freshly computed :class:`TreeRun`.
        """
        return self._cached(
            f"sst_{sample_count}_{seed}",
            lambda: self._tree_run(
                SSTPlanner(
                    self.world.occupancy,
                    self.world.bounds,
                    max_sample_count=sample_count,
                    step_size=3.4,
                    goal_tolerance=3.0,
                    witness_radius=0.45,
                    goal_bias=0.04,
                    early_stop=False,
                    seed=seed,
                ),
                sample_count,
                "SST",
            ),
        )

    def astar(self) -> GridRun:
        """Return the A* run and the cells it expanded to find the path.

        Returns:
            The cached or freshly computed :class:`GridRun`.
        """
        return self._cached("astar", self._astar_run)

    def refinement(self, sample_count: int = 7000) -> Refinement:
        """Return the prune-then-optimize chain applied to the RRT* path.

        Args:
            sample_count: Sample budget of the RRT* run being refined.

        Returns:
            The cached or freshly computed :class:`Refinement`.
        """
        return self._cached(
            f"refine_{sample_count}",
            lambda: self._refinement(sample_count),
        )

    def tracking(self, sample_count: int = 7000) -> TrackingRun:
        """Return the closed-loop run over the optimized trajectory.

        Args:
            sample_count: Sample budget of the underlying RRT* run.

        Returns:
            The cached or freshly computed :class:`TrackingRun`.
        """
        return self._cached(
            f"track_{sample_count}",
            lambda: self._tracking(sample_count),
        )

    def reachability(
        self,
        origin: Optional[np.ndarray] = None,
        profile_count: int = 17,
        horizon: float = 4.2,
        speed: float = 9.0,
    ) -> Reachability:
        """Return the vehicle model's reachable set from one pose.

        Args:
            origin: Start pose ``(x, y, heading)``.  Defaults to a pose in
                the open middle of the gallery scene.
            profile_count: Grid resolution of the turn-rate ramp sweep;
                ``profile_count ** 2`` rollouts are integrated.
            horizon: Rollout duration in seconds.
            speed: Constant speed command held over the rollout.

        Returns:
            The cached or freshly computed :class:`Reachability`.
        """
        pose = (
            np.array([120.0, 28.0, 3.90])
            if origin is None
            else np.asarray(origin, dtype=float)
        )
        key = (
            f"reach_{profile_count}_{horizon}_{speed}"
            f"_{pose[0]}_{pose[1]}_{pose[2]}"
        )
        return self._cached(
            key,
            lambda: self._reachability(pose, profile_count, horizon, speed),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _tree_run(self, planner, sample_count: int, label: str) -> TreeRun:
        """Grow a sampling planner and package its tree.

        Args:
            planner: Configured ``RRTPlanner`` or ``SSTPlanner``.
            sample_count: Budget the planner was given, kept for captions.
            label: Display name of the planner.

        Returns:
            The completed :class:`TreeRun`.
        """
        started = time.perf_counter()
        nodes, parents, path = planner.get_tree(
            self.world.start, self.world.goal
        )
        return TreeRun(
            nodes=np.asarray(nodes, dtype=float),
            parents=parents,
            path=None if path is None else np.asarray(path, dtype=float),
            sample_count=sample_count,
            seconds=time.perf_counter() - started,
            label=label,
        )

    def _astar_run(self) -> GridRun:
        """Run A* with diagnostics and convert cells to world positions.

        Returns:
            The completed :class:`GridRun`.

        Raises:
            RuntimeError: If A* finds no path in the gallery scene.
        """
        grid = self.world.grid
        started = time.perf_counter()
        path, expanded, _ = AStarPlanner(
            grid, simplify_path=False
        ).plan_with_diagnostics(self.world.start_cell, self.world.goal_cell)
        if path is None:
            raise RuntimeError("A* found no path in the gallery scene.")
        return GridRun(
            path=np.asarray([grid.position(c) for c in path], dtype=float),
            expanded=np.asarray(
                [grid.position(c) for c in expanded], dtype=float
            ),
            seconds=time.perf_counter() - started,
            cell_count=int(grid.data.size - grid.data.sum()),
        )

    def _refinement(self, sample_count: int) -> Refinement:
        """Prune and optimize the RRT* path, then resample it for stroking.

        Args:
            sample_count: Sample budget of the RRT* run to refine.

        Returns:
            The completed :class:`Refinement`.

        Raises:
            RuntimeError: If the underlying RRT* run has no solution.
        """
        run = self.rrt(sample_count)
        if run.path is None:
            raise RuntimeError("RRT* found no path to refine.")
        started = time.perf_counter()
        pruner = TrajectoryPruner(
            self.world.occupancy, step_size=np.array([3.4, 3.4])
        )
        pruned = pruner.prune([np.asarray(p) for p in run.path])
        # The pruner leaves a handful of long shortcuts.  The optimizer
        # only moves the waypoints it is given, so a sparse reference
        # optimizes into a polyline; densifying first is what lets it
        # return a curve.
        seeded = _densify(np.asarray(pruned, dtype=float), max_step=9.0)
        optimizer = TrajectoryOptimizer(
            self.world.occupancy,
            cruise_speed=9.0,
            max_speed=15.0,
            min_speed=2.5,
            weight_time=10.0,
            weight_deviation=1.0,
            weight_collision=6.0,
        )
        result = optimizer.optimize([row for row in seeded])
        states = np.asarray(result.states, dtype=float)
        durations = np.asarray(result.durations, dtype=float)
        dense, dense_speed = _resample_with_speed(states, durations)
        return Refinement(
            raw=np.asarray(run.path, dtype=float),
            pruned=np.asarray(pruned, dtype=float),
            states=states,
            durations=durations,
            dense=dense,
            dense_speed=dense_speed,
            cost=float(result.cost),
            seconds=time.perf_counter() - started,
        )

    def _tracking(self, sample_count: int) -> TrackingRun:
        """Drive a Dubins vehicle along the optimized trajectory.

        Args:
            sample_count: Sample budget of the underlying RRT* run.

        Returns:
            The completed :class:`TrackingRun`.
        """
        reference = self.refinement(sample_count).dense
        waypoints = [(float(x), float(y)) for x, y in reference]
        heading = float(
            np.arctan2(
                reference[3, 1] - reference[0, 1],
                reference[3, 0] - reference[0, 0],
            )
        )
        vehicle = DubinsVehicle(
            x=float(reference[0, 0]),
            y=float(reference[0, 1]),
            heading=heading,
            max_speed=16.0,
            max_turn_rate=0.95,
            max_acceleration=6.0,
            max_turn_rate_dot=2.5,
        )
        controller = PurePursuitController(lookahead_distance=12.0)
        loop = TrackingLoop(
            vehicle,
            controller,
            cruise_speed=9.0,
            curvature_gain=2.6,
            avoidance=ArtificialPotentialField(
                self.world.occupancy, repulsion_gain=0.7
            ),
        )
        dt = 0.08
        goal = reference[-1]
        history = []
        for _ in range(1400):
            history.append(loop.step(waypoints, dt))
            if np.linalg.norm(np.asarray(vehicle.pose[:2]) - goal) < 4.0:
                break
        poses = np.asarray([h["pose"] for h in history], dtype=float)
        carrots = np.asarray(
            [
                _find_lookahead(
                    float(pose[0]),
                    float(pose[1]),
                    waypoints,
                    _closest_index(waypoints, pose),
                    controller.lookahead_distance,
                )
                for pose in poses
            ],
            dtype=float,
        )
        return TrackingRun(
            reference=reference,
            poses=poses,
            speed=np.asarray([h["speed"] for h in history], dtype=float),
            cross_track=np.asarray(
                [h["cross_track_error"] for h in history], dtype=float
            ),
            repulsion=np.asarray(
                [h["repulsion_turn_rate"] for h in history], dtype=float
            ),
            carrots=carrots,
            dt=dt,
        )

    def _reachability(
        self,
        origin: np.ndarray,
        profile_count: int,
        horizon: float,
        speed: float,
    ) -> Reachability:
        """Integrate a sweep of turn-rate ramps through the vehicle model.

        Args:
            origin: Start pose ``(x, y, heading)``.
            profile_count: Grid resolution per ramp endpoint.
            horizon: Rollout duration in seconds.
            speed: Constant speed command held over the rollout.

        Returns:
            The completed :class:`Reachability`.
        """
        dt = 0.06
        steps = int(round(horizon / dt))
        limit = 0.55
        ramps = np.linspace(-limit, limit, profile_count)
        curves: List[np.ndarray] = []
        free: List[bool] = []
        finals: List[float] = []
        for start_rate in ramps:
            for end_rate in ramps:
                vehicle = DubinsVehicle(
                    x=float(origin[0]),
                    y=float(origin[1]),
                    heading=float(origin[2]),
                    max_speed=max(speed, 1.0),
                    min_speed=min(speed, 4.0),
                    max_turn_rate=limit,
                    max_acceleration=5.0,
                    max_turn_rate_dot=0.30,
                )
                track = [np.array([vehicle.x, vehicle.y])]
                for index in range(steps):
                    blend = index / max(steps - 1, 1)
                    rate = start_rate + (end_rate - start_rate) * blend
                    x, y, _ = vehicle.step(speed, float(rate), dt)
                    track.append(np.array([x, y]))
                curve = np.asarray(track)
                curves.append(curve)
                finals.append(float(end_rate))
                free.append(self._is_clear(curve))
        return Reachability(
            curves=curves,
            free=np.asarray(free, dtype=bool),
            turn_rate=np.asarray(finals, dtype=float),
            origin=np.asarray(origin, dtype=float),
            horizon=horizon,
        )

    def _is_clear(self, curve: np.ndarray) -> bool:
        """Return True when no sample of *curve* is inside an obstacle.

        Args:
            curve: ``(T, 2)`` rollout in world units.

        Returns:
            ``True`` when every sample is collision-free.
        """
        distances = self.world.occupancy.query_distances(curve[::3])
        return bool(np.all(distances >= self.world.clearance))

    def _cached(self, key: str, factory: Callable[[], object]):
        """Return a cached run, computing and storing it on a miss.

        Args:
            key: Cache key; also the pickle file stem.
            factory: Zero-argument callable producing the run.

        Returns:
            The cached or freshly computed run object.
        """
        path = self.cache_dir / f"{key}.pkl"
        if path.exists():
            with path.open("rb") as handle:
                return pickle.load(handle)
        value = factory()
        with path.open("wb") as handle:
            pickle.dump(value, handle)
        return value


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _densify(points: np.ndarray, max_step: float) -> np.ndarray:
    """Insert waypoints so that no segment is longer than *max_step*.

    Args:
        points: ``(N, 2)`` polyline.
        max_step: Maximum segment length in world units.

    Returns:
        The densified polyline.
    """
    out: List[np.ndarray] = [points[0]]
    for start, end in zip(points[:-1], points[1:]):
        count = max(int(np.ceil(np.linalg.norm(end - start) / max_step)), 1)
        for step in range(1, count + 1):
            out.append(start + (end - start) * (step / count))
    return np.asarray(out)


def _polyline_length(points: Optional[np.ndarray]) -> float:
    """Return the arc length of a polyline.

    Args:
        points: ``(N, 2)`` polyline, or ``None``.

    Returns:
        Arc length in world units; ``0.0`` for ``None`` or a single point.
    """
    if points is None or len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _closest_index(
    waypoints: Sequence[Tuple[float, float]], pose: np.ndarray
) -> int:
    """Return the index of the waypoint nearest to *pose*.

    Args:
        waypoints: Reference path waypoints.
        pose: Vehicle pose whose first two entries are the position.

    Returns:
        Index into *waypoints*.
    """
    array = np.asarray(waypoints, dtype=float)
    return int(np.argmin(np.linalg.norm(array - pose[:2], axis=1)))


def _resample_with_speed(
    states: np.ndarray, durations: np.ndarray, samples_per_segment: int = 40
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample optimized waypoints into a dense, speed-annotated polyline.

    A Catmull-Rom pass turns the optimizer's sparse waypoints into the
    continuous curve a vehicle would actually be asked to follow, and each
    sample carries the segment speed implied by the optimized durations.

    Args:
        states: ``(M, 2)`` optimized waypoints.
        durations: ``(M - 1,)`` optimized segment times.
        samples_per_segment: Resampling density per segment.

    Returns:
        ``(dense, speed)`` with shapes ``(P, 2)`` and ``(P,)``.
    """
    padded = np.vstack([states[0], states, states[-1]])
    points: List[np.ndarray] = []
    speeds: List[float] = []
    for index in range(len(states) - 1):
        p0, p1, p2, p3 = padded[index : index + 4]
        segment_length = float(np.linalg.norm(p2 - p1))
        duration = float(durations[index]) if index < len(durations) else 1.0
        speed = segment_length / max(duration, 1e-6)
        for t in np.linspace(0.0, 1.0, samples_per_segment, endpoint=False):
            points.append(_catmull_rom(p0, p1, p2, p3, t))
            speeds.append(speed)
    points.append(states[-1])
    speeds.append(speeds[-1])
    return np.asarray(points), np.asarray(speeds)


def _catmull_rom(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: float
) -> np.ndarray:
    """Evaluate a uniform Catmull-Rom spline segment.

    Args:
        p0: Control point before the segment.
        p1: Segment start.
        p2: Segment end.
        p3: Control point after the segment.
        t: Parameter in ``[0, 1]``.

    Returns:
        The interpolated point.
    """
    t2, t3 = t * t, t * t * t
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )


def cost_to_come(
    nodes: np.ndarray, parents: Dict[int, Optional[int]]
) -> np.ndarray:
    """Return the path cost from the root to every node of a tree.

    For the geometric planners this is the accumulated Euclidean edge
    length, which is exactly the quantity RRT* rewires to minimise — so
    colouring a tree by it shows the optimisation, not just the shape.

    Args:
        nodes: ``(N, 2)`` node positions.
        parents: Node index to parent index (``None`` at the root).

    Returns:
        ``(N,)`` array of costs; unreachable nodes get ``nan``.
    """
    cost = np.full(len(nodes), np.nan)
    order = sorted(range(len(nodes)), key=lambda i: _depth(parents, i))
    for index in order:
        parent = parents.get(index)
        if parent is None:
            cost[index] = 0.0
        elif parent < len(nodes) and not np.isnan(cost[parent]):
            cost[index] = cost[parent] + float(
                np.linalg.norm(nodes[index] - nodes[parent])
            )
    filled = np.nanmax(cost) if np.any(~np.isnan(cost)) else 0.0
    return np.where(np.isnan(cost), filled, cost)


def _depth(parents: Dict[int, Optional[int]], index: int) -> int:
    """Return the hop distance from the root for one node.

    Args:
        parents: Node index to parent index.
        index: Node index to measure.

    Returns:
        Number of hops to the root.
    """
    hops, cursor, guard = 0, parents.get(index), 0
    while cursor is not None and guard < len(parents) + 1:
        hops += 1
        cursor = parents.get(cursor)
        guard += 1
    return hops
