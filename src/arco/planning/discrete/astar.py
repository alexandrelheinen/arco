from __future__ import annotations

import heapq
from typing import Any, Callable, Dict, List, Optional, Tuple

from .base import DiscretePlanner



class AStarPlanner(DiscretePlanner):
    """
    A* path planner for general graphs (including grids).
    Accepts any Graph (e.g., Grid, Occupancy, custom Graph).
    """

    heuristic: Callable[[Tuple[int, ...], Tuple[int, ...]], float]

    def __init__(
        self,
        graph: Any,
        heuristic: Optional[Callable[[Tuple[int, ...], Tuple[int, ...]], float]] = None,
    ) -> None:
        super().__init__(graph)
        self.heuristic = heuristic if heuristic is not None else graph.distance

    def plan(
        self, start: Tuple[int, ...], goal: Tuple[int, ...]
    ) -> Optional[List[Tuple[int, ...]]]:
        open_set: List[Tuple[float, Tuple[int, ...]]] = []
        heapq.heappush(open_set, (0, start))
        came_from: Dict[Tuple[int, ...], Tuple[int, ...]] = {}
        g_score: Dict[Tuple[int, ...], float] = {start: 0}
        f_score: Dict[Tuple[int, ...], float] = {start: self.heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                return self._reconstruct_path(came_from, current)
            for neighbor in self.graph.neighbors(current):
                # If the graph supports is_occupied, skip occupied nodes
                if hasattr(self.graph, 'is_occupied') and self.graph.is_occupied(neighbor):
                    continue
                tentative_g = g_score[current] + self.graph.distance(current, neighbor)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None  # No path found

    def _reconstruct_path(
        self,
        came_from: Dict[Tuple[int, ...], Tuple[int, ...]],
        current: Tuple[int, ...],
    ) -> List[Tuple[int, ...]]:
        path: List[Tuple[int, ...]] = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
