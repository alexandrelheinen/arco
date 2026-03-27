"""
Mapping module for spatial representations.

This module provides base classes for grid and occupancy maps, following the ARCO architecture guidelines.
"""

from .grid.base import Grid
from .grid.manhattan import ManhattanGrid
from .grid.euclidean import EuclideanGrid
from .occupancy import Occupancy
