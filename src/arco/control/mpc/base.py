"""MPCTracker: abstract base for receding-horizon path trackers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from arco._arco import DubinsPathFollowingMPC
from arco.control.mpc.result import MPCStepResult


class MPCTracker(ABC):
    """Abstract base for multi-state model-predictive path trackers.

    Unlike the scalar :class:`~arco.control.mpc.controller.MPCController`
    stub, this interface operates on SE(2) (or joint-space) poses and
    returns structured :class:`MPCStepResult` diagnostics.
    """

    @abstractmethod
    def set_reference(self, waypoints: Sequence[tuple[float, float]]) -> None:
        """Set or replace the reference path.

        Args:
            waypoints: Ordered ``(x, y)`` waypoints in world frame.
        """

    @abstractmethod
    def step(
        self,
        pose: tuple[float, float, float],
        *,
        speed: float,
        turn_rate: float,
        dt: float,
    ) -> MPCStepResult:
        """Compute one receding-horizon control step.

        Args:
            pose: Current vehicle pose ``(x, y, heading)``.
            speed: Current forward speed (m/s).
            turn_rate: Current turn rate (rad/s).
            dt: Control period until the next call (s).

        Returns:
            Structured command and diagnostics.
        """


# The compiled controller cannot inherit from a Python abstract base, so
# it is registered as a virtual subclass instead: an ``isinstance`` check
# against this interface keeps answering True, which is what a caller
# written against the protocol relies on.
MPCTracker.register(DubinsPathFollowingMPC)
