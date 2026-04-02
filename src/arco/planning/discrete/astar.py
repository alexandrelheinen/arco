"""A* discrete path planner."""

from __future__ import annotations

import heapq
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from .base import DiscretePlanner

logger = logging.getLogger(__name__)


class AStarPlanner(DiscretePlanner):
    """
    A* path planner for general graphs (including grids).
    Accepts any Graph (e.g., Grid, Occupancy, custom Graph).

    Tie-breaking: when multiple nodes share the same f-score the planner
    uses the heuristic value h as a secondary key (prefer the node closer
    to the goal) and a monotonically increasing insertion counter as a
    tertiary key (FIFO among equal h).  This avoids lexicographic
    comparison of node tuples, which caused systematic L-shaped paths on
    symmetric grids.

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
        elif hasattr(graph, "heuristic"):
            self.heuristic = graph.heuristic
        else:
            self.heuristic = graph.distance

    def plan(self, start: Any, goal: Any) -> Optional[List[Any]]:
        """Plan a path from start to goal using A*.

        Args:
            start: The start node.
            goal: The goal node.

        Returns:
            A list of nodes from start to goal, or None if no path exists.
        """
        logger.debug("A* plan: start=%s goal=%s", start, goal)
        open_set: List[Tuple[float, float, int, Any]] = []
        counter: int = 0
        h_start = self.heuristic(start, goal)
        f_start = 0.0 + h_start  # g(start)=0, so f=h
        heapq.heappush(open_set, (f_start, h_start, counter, start))
        came_from: Dict[Any, Any] = {}
        g_score: Dict[Any, float] = {start: 0}

        while open_set:
            _, _, _, current = heapq.heappop(open_set)
            if current == goal:
                path = self._reconstruct_path(came_from, current)
                logger.debug("A* found path: length=%d", len(path))
                return path
            for neighbor in self.graph.neighbors(current):
                # If the graph supports is_occupied, skip occupied nodes
                if hasattr(
                    self.graph, "is_occupied"
                ) and self.graph.is_occupied(neighbor):
                    continue
                tentative_g = g_score[current] + self.graph.distance(
                    current, neighbor
                )
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h = self.heuristic(neighbor, goal)
                    f = tentative_g + h
                    counter += 1
                    heapq.heappush(open_set, (f, h, counter, neighbor))
        logger.debug("A* found no path from %s to %s", start, goal)
        return None  # No path found

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
