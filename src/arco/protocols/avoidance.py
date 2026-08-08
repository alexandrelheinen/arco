"""AvoidanceStrategy: duck-typed contract for reactive obstacle avoidance."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class AvoidanceStrategy(Protocol):
    """Reactive avoidance correction used by path-tracking loops.

    ``__call__(x, y, theta) → turn_rate_bias`` returns an additive turn-rate
    correction (rad/s).  The default APF strategy returns ``0.0`` when
    disabled or when no obstacle is nearby.
    """

    def __call__(self, x: float, y: float, theta: float) -> float:
        """Return an additive turn-rate bias for pose ``(x, y, theta)``."""
