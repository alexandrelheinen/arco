"""A* discrete path planner."""

from __future__ import annotations

import heapq
import logging
from collections.abc import Sequence
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .base import DiscretePlanner

logger = logging.getLogger(__name__)


class AStarPlanner(DiscretePlanner):
    """
    A* path planner for general graphs (including grids).
    Accepts any Graph (e.g., Grid, Occupancy, custom Graph).

    Tie-breaking: when multiple nodes share the same f-score the planner
    first prefers moves that continue the current direction (fewer heading
    changes), then uses the heuristic value h as a secondary key (prefer
    the node closer to the goal), and finally a monotonically increasing
    insertion counter (FIFO among equal h). This avoids lexicographic
    comparison of node tuples and reduces zig-zagging in symmetric grids.

    Default heuristic: ``graph.heuristic`` if the graph exposes it,
    otherwise ``graph.distance``.  Grid subclasses expose a Euclidean
    ``heuristic`` which is admissible on both Manhattan and Euclidean
    grids and guides A* toward diagonal/staircase paths instead of the
    L-shape that arises when all f-scores are tied.
    """

    heuristic: Callable[[Any, Any], float]

    def __init__(
        self,
        graph: Any,
        heuristic: Optional[Callable[[Any, Any], float]] = None,
    ) -> None:
        """Initialize AStarPlanner.

        Args:
            graph: The graph or grid to plan on. Must expose ``neighbors``
                and ``distance`` methods. Optionally exposes ``heuristic``
                and ``is_occupied``.
            heuristic: Optional heuristic callable ``(node, goal) -> float``.
                Defaults to ``graph.heuristic`` if available, else
                ``graph.distance``.
        """
        super().__init__(graph)
        if heuristic is not None:
            self.heuristic = heuristic
            logger.debug("Setting custom heuristic")
        elif hasattr(graph, "heuristic"):
            self.heuristic = graph.heuristic
            logger.debug("Using graph-provided heuristic")
        else:
            self.heuristic = graph.distance
            logger.debug("Using graph distance as heuristic")

    def plan(self, start: Any, goal: Any) -> Optional[List[Any]]:
        """Plan a path from start to goal using A*.

        Args:
            start: The start node.
            goal: The goal node.

        Returns:
            A list of nodes from start to goal, or None if no path exists.
        """
        path, _, _ = self.plan_with_diagnostics(start, goal)
        return path

    def plan_with_diagnostics(
        self,
        start: Any,
        goal: Any,
    ) -> tuple[Optional[List[Any]], List[Any], Dict[Any, Any]]:
        """Plan using A* and return path plus exploration diagnostics.

        The returned diagnostics allow visualizers to reconstruct the A*
        exploration tree without re-running the algorithm.

        Args:
            start: The start node.
            goal: The goal node.

        Returns:
            ``(path, expanded_order, parent_map)`` where:

            - ``path`` is the simplified path from ``start`` to ``goal``, or
              ``None`` if no path exists.
            - ``expanded_order`` is the ordered list of nodes popped from the
              frontier and expanded.
            - ``parent_map`` maps discovered nodes to their predecessor.
        """
        logger.debug("A* plan: start=%s goal=%s", start, goal)
        open_set: List[Tuple[float, int, float, int, Any]] = []
        counter: int = 0
        expanded_count = 0
        h_start = self.heuristic(start, goal)
        f_start = 0.0 + h_start  # g(start)=0, so f=h
        heapq.heappush(open_set, (f_start, 0, h_start, counter, start))
        came_from: Dict[Any, Any] = {}
        g_score: Dict[Any, float] = {start: 0}
        direction_rank: Dict[Any, int] = {start: 0}
        expanded_order: List[Any] = []
        closed_set: set[Any] = set()

        while open_set:
            _, _, _, _, current = heapq.heappop(open_set)
            if current in closed_set:
                continue
            closed_set.add(current)
            expanded_order.append(current)
            expanded_count += 1
            if current == goal:
                path = self._reconstruct_path(came_from, current)
                simplified_path = self._simplify_path(path)
                logger.debug(
                    (
                        "A*: finished success raw_length=%d simplified_length=%d "
                        "expanded=%d frontier=%d"
                    ),
                    len(path),
                    len(simplified_path),
                    expanded_count,
                    len(open_set),
                )
                return simplified_path, expanded_order, came_from
            for neighbor in self.graph.neighbors(current):
                # If the graph supports is_occupied, skip occupied nodes
                if hasattr(
                    self.graph, "is_occupied"
                ) and self.graph.is_occupied(neighbor):
                    continue
                tentative_g = g_score[current] + self.graph.distance(
                    current, neighbor
                )
                direction_change_penalty = self._turn_penalty(
                    current,
                    neighbor,
                    came_from,
                )
                better_cost = (
                    neighbor not in g_score
                    or tentative_g < g_score[neighbor] - 1e-12
                )
                equal_cost_better_direction = (
                    neighbor in g_score
                    and abs(tentative_g - g_score[neighbor]) <= 1e-12
                    and direction_change_penalty
                    < direction_rank.get(neighbor, 1)
                )
                if better_cost or equal_cost_better_direction:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    direction_rank[neighbor] = direction_change_penalty
                    h = self.heuristic(neighbor, goal)
                    f = tentative_g + h
                    counter += 1
                    heapq.heappush(
                        open_set,
                        (
                            f,
                            direction_change_penalty,
                            h,
                            counter,
                            neighbor,
                        ),
                    )
        logger.debug(
            (
                "A*: finished no-path start=%s goal=%s expanded=%d "
                "frontier=%d"
            ),
            start,
            goal,
            expanded_count,
            len(open_set),
        )
        return None, expanded_order, came_from

    def _reconstruct_path(
        self,
        came_from: Dict[Any, Any],
        current: Any,
    ) -> List[Any]:
        """Reconstruct the path from start to current by following came_from.

        Args:
            came_from: A mapping from each node to its predecessor.
            current: The goal node to trace back from.

        Returns:
            A list of nodes from start to current.
        """
        path: List[Any] = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    @staticmethod
    def _as_direction_vector(node: Any) -> np.ndarray | None:
        """Return node as a numeric vector, or ``None`` if unsupported."""
        if isinstance(node, np.ndarray):
            arr = np.asarray(node, dtype=float).reshape(-1)
            return arr if arr.size >= 2 else None
        if isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
            try:
                arr = np.asarray(node, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                return None
            return arr if arr.size >= 2 else None
        return None

    def _simplify_path(self, path: List[Any]) -> List[Any]:
        """Simplify a path by collapsing consecutive steps with same direction.

        This mandatory in-algorithm simplification is distinct from the
        trajectory-pruning stage used later in the planning pipeline.
        """
        if len(path) < 3:
            return path

        first = self._as_direction_vector(path[0])
        second = self._as_direction_vector(path[1])
        if first is None or second is None:
            return path

        def _direction(a: Any, b: Any) -> tuple[float, ...] | None:
            va = self._as_direction_vector(a)
            vb = self._as_direction_vector(b)
            if va is None or vb is None or va.shape != vb.shape:
                return None
            delta = vb - va
            norm = float(np.linalg.norm(delta))
            if norm <= 1e-12:
                return None
            return tuple(np.round(delta / norm, 8).tolist())

        prev_dir = _direction(path[0], path[1])
        if prev_dir is None:
            return path

        simplified: List[Any] = [path[0]]
        for idx in range(1, len(path) - 1):
            curr_dir = _direction(path[idx], path[idx + 1])
            if curr_dir is None:
                return path
            if curr_dir != prev_dir:
                simplified.append(path[idx])
                prev_dir = curr_dir
        simplified.append(path[-1])
        return simplified

    def _turn_penalty(
        self,
        current: Any,
        neighbor: Any,
        came_from: Dict[Any, Any],
    ) -> int:
        """Return tie-break penalty: 0 keeps direction, 1 changes it.

        This value is used only in the priority-queue sort key (not in
        ``g``), so path optimality is preserved.
        """
        parent = came_from.get(current)
        if parent is None:
            return 0

        prev_a = self._as_direction_vector(parent)
        prev_b = self._as_direction_vector(current)
        next_b = self._as_direction_vector(neighbor)
        if prev_a is None or prev_b is None or next_b is None:
            return 0
        if prev_a.shape != prev_b.shape or prev_b.shape != next_b.shape:
            return 0

        prev_dir = prev_b - prev_a
        next_dir = next_b - prev_b
        if (
            float(np.linalg.norm(prev_dir)) <= 1e-12
            or float(np.linalg.norm(next_dir)) <= 1e-12
        ):
            return 0

        prev_key = tuple(np.round(prev_dir / np.linalg.norm(prev_dir), 8))
        next_key = tuple(np.round(next_dir / np.linalg.norm(next_dir), 8))
        return 0 if prev_key == next_key else 1
