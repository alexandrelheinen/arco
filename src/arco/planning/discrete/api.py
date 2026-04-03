"""Public API wrappers for discrete grid planners."""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from arco.mapping import EuclideanGrid, ManhattanGrid

from .astar import AStarPlanner


class AStar:
    """Public API wrapper for the A* planner.

    Accepts a numpy grid (0=free, 1=occupied).
    By default uses ManhattanGrid; pass grid_type='euclidean' for EuclideanGrid.
    """

    def __init__(self, grid: np.ndarray, grid_type: str = "manhattan") -> None:
        """Initialize the AStar wrapper.

        Args:
            grid: A numpy array where 0=free and 1=occupied.
            grid_type: Grid connectivity type. Use 'euclidean' for diagonal
                neighbors or 'manhattan' (default) for axis-aligned neighbors.
        """
        if grid_type == "euclidean":
            self._grid = EuclideanGrid(grid.shape)
        else:
            self._grid = ManhattanGrid(grid.shape)
        self._grid.data = np.array(grid, dtype=np.uint8)
        self._planner = AStarPlanner(self._grid)

    def search(self, start: Any, goal: Any) -> Optional[List[Any]]:
        """Search for a path from start to goal.

        Args:
            start: The start node index.
            goal: The goal node index.

        Returns:
            A list of node indices from start to goal, or None if no path exists.
        """
        return self._planner.plan(start, goal)


class DStarLite:
    """Public API wrapper for D* planner (stub — not yet implemented)."""

    def __init__(self, grid: np.ndarray) -> None:
        """Initialize the DStarLite wrapper.

        Args:
            grid: A numpy array where 0=free and 1=occupied.
        """

    def search(self, start: Any, goal: Any) -> Optional[List[Any]]:
        """Search for a path from start to goal (not yet implemented).

        Args:
            start: The start node index.
            goal: The goal node index.

        Returns:
            A list of node indices from start to goal, or None if no path exists.

        Raises:
            NotImplementedError: D* planner is not yet implemented.
        """
        raise NotImplementedError("D* planner not yet implemented.")
