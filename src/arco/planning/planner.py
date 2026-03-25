class Planner:
    """Base class for path planners."""

    def __init__(self, graph):
        self._graph = graph


class DStarPlanner(Planner):
    """
    D* path planner for dynamic replanning (stub).
    """

    def __init__(self, grid):
        super().__init__(grid)
        self.grid = grid

    def plan(self, start, goal):
        raise NotImplementedError("D* planner not yet implemented.")


class RRTPlanner(Planner):
    """
    RRT planner for continuous state spaces (stub).
    """

    def __init__(self, occupancy):
        super().__init__(occupancy)
        self.occupancy = occupancy

    def plan(self, start, goal):
        raise NotImplementedError("RRT planner not yet implemented.")


class SSTPlanner(Planner):
    """
    Stable Sparse RRT planner for kinodynamic planning (stub).
    """

    def __init__(self, occupancy):
        super().__init__(occupancy)
        self.occupancy = occupancy

    def plan(self, start, goal):
        raise NotImplementedError("SST planner not yet implemented.")


"""Planner module for planning problems."""

from ..mapping.graph import Graph


class Planner:
    """Base class for path planners."""

    def __init__(self, graph: Graph):
        """Initialize the planner with a graph."""
        self._graph = graph


class AStarPlanner(Planner):
    """
    A* path planner for Grid maps (Manhattan or Euclidean).
    Uses the grid's distance and neighbor logic.
    """

    def __init__(self, grid, heuristic=None):
        super().__init__(grid)
        self.grid = grid
        # Use grid's distance as default heuristic
        self.heuristic = heuristic if heuristic is not None else grid.distance

    def plan(self, start, goal):
        import heapq

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
