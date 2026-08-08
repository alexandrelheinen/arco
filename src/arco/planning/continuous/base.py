"""ContinuousPlanner: base class for continuous-space planners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional

import numpy as np

from arco.mapping.occupancy import Occupancy
from arco.planning.cost import PlannerCost


class ContinuousPlanner(PlannerCost, ABC):
    """Base class for planners operating in continuous state spaces.

    Inherits :meth:`~arco.planning.cost.PlannerCost.distance` and
    :meth:`~arco.planning.cost.PlannerCost.heuristic` from
    :class:`~arco.planning.cost.PlannerCost`.  :meth:`distance` is
    overridden to use step-size-normalized Euclidean distance once a
    subclass sets :attr:`step_size`.

    Subclasses must implement :meth:`plan`.
    """

    def __init__(self, occupancy: Occupancy) -> None:
        """Initialize the planner with an occupancy map.

        Args:
            occupancy: The occupancy map for collision checking.
        """
        self.occupancy = occupancy

    def distance(self, state_a: Any, state_b: Any) -> float:
        """Return step-size-normalized Euclidean distance.

        Each axis is divided by the corresponding entry of
        :attr:`step_size` (default ``1.0``) before the L2 norm is taken,
        so mixed units across dimensions are handled uniformly.

        Args:
            state_a: Origin state.
            state_b: Destination state.

        Returns:
            Non-negative normalized distance.
        """
        step = np.asarray(getattr(self, "step_size", 1.0), dtype=float)
        a = np.asarray(state_a, dtype=float).reshape(-1)
        b = np.asarray(state_b, dtype=float).reshape(-1)
        return float(np.linalg.norm((b - a) / step))

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
