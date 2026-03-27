
"""Occupancy: Abstract base for continuous occupancy maps."""

from __future__ import annotations

from abc import abstractmethod
from typing import Tuple
from arco.mapping.graph import Graph


class Occupancy(Graph):
    """
    Abstract base for continuous occupancy maps (for RRT, SST, etc).

    Inherits from Graph, so planners can treat occupancy maps as graphs.
    Subclasses may use point clouds, kd-trees, etc.
    Provides a unified interface for obstacle queries in continuous space.
    """

    @abstractmethod
    def is_occupied(self, point: Tuple[float, ...]) -> bool:
        """Return True if the given point is occupied."""
        pass
