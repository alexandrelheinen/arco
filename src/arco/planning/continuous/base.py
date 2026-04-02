"""ContinuousPlanner: base class for continuous-space planners."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, List, Optional

from arco.mapping.occupancy import Occupancy


class ContinuousPlanner:
    """Base class for planners operating in continuous state spaces."""

    def __init__(self, occupancy: Occupancy) -> None:
        """Initialize the planner with an occupancy map.

        Args:
            occupancy: The occupancy map for collision checking.
        """
        self.occupancy = occupancy

    @abstractmethod
    def plan(self, start: Any, goal: Any) -> Optional[List[Any]]:
        """Plan a path from start to goal.

        Args:
            start: The start state.
            goal: The goal state.

        Returns:
            A list of states from start to goal, or None if no path exists.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
