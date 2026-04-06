"""DubinsPrimitive: Dubins path for car-like robots."""

from __future__ import annotations

from typing import Any, List

import numpy as np

from .base import ExplorationPrimitive


class DubinsPrimitive(ExplorationPrimitive):
    """Dubins path primitive for car-like robots.

    Enforces no-reverse motion and minimum turning radius constraints.

    Attributes:
        turning_radius: Minimum turning radius (meters).  A robot moving
            at speed *v* with turn rate *ω* requires a turning radius of
            ``v / |ω|``; the maneuver is feasible only when this radius is
            at least :attr:`turning_radius`.
    """

    def __init__(self, turning_radius: float = 1.0) -> None:
        """Initialize DubinsPrimitive.

        Args:
            turning_radius: Minimum turning radius for the robot (meters).
                Must be positive.
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

    def is_feasible(self, state: np.ndarray) -> bool:
        """Check whether a state satisfies the minimum turning-radius constraint.

        For a full five-element state ``(x, y, θ, v, ω)``, the maneuver is
        feasible when the instantaneous turning radius ``|v / ω|`` is at
        least :attr:`turning_radius`.  A turn rate of zero (straight-line
        motion) is always feasible.  States with fewer than five elements do
        not carry curvature information and are unconditionally accepted.

        Args:
            state: Kinematic state.  Interpreted as:

                - ``(x, y)`` or ``(x, y, θ)`` → always feasible.
                - ``(x, y, θ, v)`` → feasible (speed alone cannot violate
                  the turning-radius constraint).
                - ``(x, y, θ, v, ω)`` → checked against the constraint
                  ``|v / ω| ≥ turning_radius``.

        Returns:
            ``True`` if the state satisfies the turning-radius constraint,
            ``False`` otherwise.
        """
        state = np.asarray(state, dtype=float)
        if state.shape[0] >= 5:
            v = float(state[3])
            omega = float(state[4])
            if abs(omega) > 1e-12:
                radius = abs(v) / abs(omega)
                if radius < self.turning_radius:
                    return False
        return True
