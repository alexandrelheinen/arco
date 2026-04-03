"""KDTreeOccupancy: sparse continuous occupancy map backed by a KD-tree."""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import numpy as np
from scipy.spatial import KDTree

from .occupancy import Occupancy


class KDTreeOccupancy(Occupancy):
    """Continuous, sparse occupancy map backed by a KD-tree.

    Stores only the obstacle points (no empty-space cells) and uses a
    :class:`scipy.spatial.KDTree` for efficient nearest-neighbour lookup.
    Collision checking is done by comparing the distance to the nearest
    obstacle against a configurable clearance radius.

    Args:
        points: Obstacle point coordinates.  Each row is one obstacle
            point.  Accepts any array-like with shape ``(N, D)`` where
            *D* is the spatial dimension.
        clearance: Minimum safe distance from any obstacle point.  A
            query point is considered occupied when its distance to the
            nearest obstacle is strictly less than this value.
    """

    def __init__(
        self,
        points: Union[np.ndarray, List[Sequence[float]]],
        clearance: float = 0.5,
    ) -> None:
        """Initialize the KDTreeOccupancy.

        Args:
            points: Obstacle point coordinates as an array-like of shape
                ``(N, D)``.  Must contain at least one point.
            clearance: Minimum safe distance from any obstacle.  Defaults
                to 0.5 world-units.

        Raises:
            ValueError: If *points* is empty or *clearance* is not positive.
        """
        super().__init__()
        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        if pts.shape[0] == 0:
            raise ValueError("points must contain at least one obstacle.")
        if clearance <= 0:
            raise ValueError(f"clearance must be positive, got {clearance!r}.")
        self._points = pts
        self._tree = KDTree(pts)
        self.clearance = clearance

    # ------------------------------------------------------------------
    # Occupancy interface
    # ------------------------------------------------------------------

    def nearest_obstacle(self, point: np.ndarray) -> Tuple[float, np.ndarray]:
        """Return the distance and position of the nearest obstacle point.

        Args:
            point: Query position as a numpy array of shape ``(D,)``.

        Returns:
            A ``(distance, nearest_point)`` tuple where *distance* is the
            Euclidean distance to the nearest obstacle and *nearest_point*
            is its coordinates as a numpy array.
        """
        pt = np.asarray(point, dtype=float)
        dist, idx = self._tree.query(pt)
        return float(dist), self._points[idx].copy()

    def is_occupied(self, point: np.ndarray) -> bool:
        """Return True if *point* is within the clearance radius of an obstacle.

        Args:
            point: Query position as a numpy array of shape ``(D,)``.

        Returns:
            True if the nearest obstacle is closer than :attr:`clearance`,
            False otherwise.
        """
        dist, _ = self.nearest_obstacle(point)
        return dist < self.clearance

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def points(self) -> np.ndarray:
        """Obstacle points array of shape ``(N, D)``."""
        return self._points.copy()

    @property
    def dimension(self) -> int:
        """Spatial dimension of the obstacle space."""
        return int(self._points.shape[1])

    def query_distances(self, points: np.ndarray) -> np.ndarray:
        """Return the distance from each query point to its nearest obstacle.

        Batch counterpart of :meth:`nearest_obstacle` — uses a single
        KD-tree query for efficiency when many points are queried at once.

        Args:
            points: Query positions as an array of shape ``(M, D)``.

        Returns:
            Distance array of shape ``(M,)`` where entry *i* is the
            Euclidean distance from ``points[i]`` to the nearest obstacle.
        """
        distances, _ = self._tree.query(np.asarray(points, dtype=float))
        return np.asarray(distances, dtype=float)
