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

    Optional clearance / batch-distance hooks default to conservative
    fallbacks so pruner and optimizer code can call them without
    ``getattr`` / ``hasattr`` branching.
    """

    @property
    def clearance(self) -> float:
        """Minimum free margin around obstacles (meters).

        Defaults to ``0.0`` (binary occupancy only).  Concrete maps such as
        :class:`~arco.mapping.kdtree.KDTreeOccupancy` override this.
        """
        return 0.0

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

    def query_distances(self, points: np.ndarray) -> np.ndarray:
        """Return nearest-obstacle distances for each row of *points*.

        Default implementation loops over :meth:`nearest_obstacle`.
        Subclasses may override with a batch query.

        Args:
            points: Array of shape ``(N, D)`` query positions.

        Returns:
            Array of shape ``(N,)`` distances.
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        return np.asarray(
            [self.nearest_obstacle(p)[0] for p in pts], dtype=float
        )

    def segment_free(
        self,
        a: np.ndarray,
        b: np.ndarray,
        *,
        sample_count: int = 12,
    ) -> bool:
        """Return True if the segment from *a* to *b* is collision-free.

        Default policy matches historical RRT*/SST checking: linspace
        *sample_count* points and test each with :meth:`is_occupied`.

        Args:
            a: Segment start.
            b: Segment end.
            sample_count: Number of sample points including endpoints.

        Returns:
            True when every sample is free.
        """
        count = max(int(sample_count), 2)
        for t in np.linspace(0.0, 1.0, count):
            pt = a + t * (b - a)
            if self.is_occupied(pt):
                return False
        return True
