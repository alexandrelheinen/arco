"""OccupancyLike: duck-typed contract for continuous collision maps."""

from __future__ import annotations

from typing import Protocol, Tuple, runtime_checkable

import numpy as np


@runtime_checkable
class OccupancyLike(Protocol):
    """Continuous occupancy surface used by RRT*/SST/pruner/optimizer.

    Required methods match :class:`~arco.mapping.occupancy.Occupancy`.
    Optional attributes ``clearance`` and ``query_distances`` are used by
    pruner/optimizer when available.
    """

    def nearest_obstacle(self, point: np.ndarray) -> Tuple[float, np.ndarray]:
        """Return ``(distance, nearest_point)`` for the closest obstacle."""

    def is_occupied(self, point: np.ndarray) -> bool:
        """Return True if *point* is in collision."""
