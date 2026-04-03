"""ContinuousPlanner: base class for continuous-space planners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional

import numpy as np

from arco.mapping.occupancy import Occupancy


class ContinuousPlanner(ABC):
    """Base class for planners operating in continuous state spaces.

    Subclasses must implement :meth:`plan`.
    """

    def __init__(self, occupancy: Occupancy) -> None:
        """Initialize the planner with an occupancy map.

        Args:
            occupancy: The occupancy map for collision checking.
        """
        self.occupancy = occupancy

    @abstractmethod
    def plan(
        self, start: np.ndarray, goal: np.ndarray
    ) -> Optional[List[np.ndarray]]:
        """Plan a path from start to goal.

        Args:
            start: The start state as a numpy array.
            goal: The goal state as a numpy array.

        Returns:
            A list of numpy arrays from start to goal, or None if no path
            exists.
        """
