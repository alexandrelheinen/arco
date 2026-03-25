import heapq

from .base import DiscretePlanner


class AStarPlanner(DiscretePlanner):
    """
    A* path planner for grid-based maps.
    """

    def __init__(self, grid, heuristic=None):
        super().__init__(grid)
        self.heuristic = heuristic if heuristic is not None else grid.distance

    def plan(self, start, goal):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                return self._reconstruct_path(came_from, current)
            for neighbor in self.grid.neighbors(current):
                if self.grid.is_occupied(neighbor):
                    continue
                tentative_g = g_score[current] + self.grid.distance(current, neighbor)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None  # No path found

    def _reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
