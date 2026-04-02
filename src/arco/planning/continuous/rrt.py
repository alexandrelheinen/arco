"""RRTPlanner: Rapidly-exploring Random Tree planner (stub)."""

from __future__ import annotations

from typing import Any, List, Optional

from arco.mapping.occupancy import Occupancy

from .base import ContinuousPlanner


class RRTPlanner(ContinuousPlanner):
    """RRT planner for continuous state spaces (stub)."""

    def __init__(self, occupancy: Occupancy) -> None:
        """Initialize RRTPlanner.

        Args:
            occupancy: The occupancy map for collision checking.
        """
        super().__init__(occupancy)

    def plan(self, start: Any, goal: Any) -> Optional[List[Any]]:
        """Plan a path using RRT (not yet implemented).

        Args:
            start: The start state.
            goal: The goal state.

        Returns:
            A list of states from start to goal, or None if no path exists.

        Raises:
            NotImplementedError: RRT planner is not yet implemented.
        """
        raise NotImplementedError("RRT planner not yet implemented.")
