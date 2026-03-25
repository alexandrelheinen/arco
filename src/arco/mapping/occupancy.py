"""
Occupancy: Abstract base for continuous occupancy maps.
"""

from abc import ABC, abstractmethod


class Occupancy(ABC):
    """
    Abstract base for continuous occupancy maps (for RRT, SST, etc).
    Subclasses may use point clouds, kd-trees, etc.
    Provides a unified interface for obstacle queries in continuous space.
    """

    @abstractmethod
    def is_occupied(self, point):
        """Return True if the given point is occupied."""
        pass
