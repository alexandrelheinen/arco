"""ExplorationPrimitive: abstract base for exploration primitives."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List


class ExplorationPrimitive(ABC):
    """Abstract base for exploration primitives (e.g., Dubins, Reeds-Shepp).

    Used in RRT-based planners to ensure kinematic feasibility.
    """

    @abstractmethod
    def steer(self, from_state: Any, to_state: Any) -> List[Any]:
        """Return a feasible path segment from from_state to to_state.

        Args:
            from_state: The starting state.
            to_state: The target state.

        Returns:
            A list of states representing the path segment.
        """
        pass
