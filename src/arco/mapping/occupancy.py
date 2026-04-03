"""Occupancy: Abstract base for continuous occupancy maps."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from arco.mapping.graph import Graph


class Occupancy(Graph, ABC):
    """Abstract base for continuous occupancy maps (for RRT, SST, etc).

    Inherits from Graph, so planners can treat occupancy maps as graphs.
    Subclasses may use point clouds, kd-trees, etc.
    Provides a unified interface for obstacle queries in continuous space.
    """

    @abstractmethod
    def nearest_obstacle(self, point: np.ndarray) -> Tuple[float, np.ndarray]:
        """Return the distance and coordinates of the nearest obstacle.

        Args:
            point: Query position as a numpy array.

        Returns:
            A ``(distance, nearest_point)`` tuple where *distance* is the
            Euclidean distance to the nearest obstacle and *nearest_point*
            is its coordinates as a numpy array.
        """

    @abstractmethod
    def is_occupied(self, point: np.ndarray) -> bool:
        """Return True if the given point is in collision.

        Args:
            point: The coordinates to check as a numpy array.

        Returns:
            True if the point is occupied, False otherwise.
        """
