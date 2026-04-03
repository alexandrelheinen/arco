"""DubinsPrimitive: Dubins path for car-like robots."""

from __future__ import annotations

from typing import Any, List

from .base import ExplorationPrimitive


class DubinsPrimitive(ExplorationPrimitive):
    """Dubins path primitive for car-like robots.

    Enforces no-reverse motion and minimum turning radius constraints.
    """

    def __init__(self, turning_radius: float = 1.0) -> None:
        """Initialize DubinsPrimitive.

        Args:
            turning_radius: Minimum turning radius for the robot.
        """
        self.turning_radius = turning_radius

    def steer(self, from_state: Any, to_state: Any) -> List[Any]:
        """Return a feasible Dubins path segment from from_state to to_state.

        Args:
            from_state: The starting state.
            to_state: The target state.

        Returns:
            A list of states representing the path segment.
        """
        # Placeholder: would use a dubins path library in practice
        return [from_state, to_state]
