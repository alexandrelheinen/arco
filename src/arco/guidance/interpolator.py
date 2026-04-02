"""Interpolator: abstract base for path interpolation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List


class Interpolator(ABC):
    """Abstract base for interpolation (e.g., B-splines, shortcutting).

    Used to convert discrete node sequences to continuous trajectories.
    """

    @abstractmethod
    def interpolate(self, path: List[Any]) -> List[Any]:
        """Return a continuous trajectory from a discrete path.

        Args:
            path: A list of discrete waypoints.

        Returns:
            A list of waypoints representing the interpolated trajectory.
        """
        pass
