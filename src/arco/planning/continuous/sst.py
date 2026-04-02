"""SSTPlanner: Stable Sparse RRT planner for kinodynamic planning (stub)."""

from __future__ import annotations

from typing import Any, List, Optional

from arco.mapping.occupancy import Occupancy

from .base import ContinuousPlanner


class SSTPlanner(ContinuousPlanner):
    """Stable Sparse RRT planner for kinodynamic planning (stub)."""

    def __init__(self, occupancy: Occupancy) -> None:
        """Initialize SSTPlanner.

        Args:
            occupancy: The occupancy map for collision checking.
        """
        super().__init__(occupancy)

    def plan(self, start: Any, goal: Any) -> Optional[List[Any]]:
        """Plan a path using SST (not yet implemented).

        Args:
            start: The start state.
            goal: The goal state.

        Returns:
            A list of states from start to goal, or None if no path exists.

        Raises:
            NotImplementedError: SST planner is not yet implemented.
        """
        raise NotImplementedError("SST planner not yet implemented.")
