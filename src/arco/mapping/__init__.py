"""
Mapping module for spatial representations.

This module provides base classes for grid and occupancy maps, following the ARCO architecture guidelines.
"""

from .grid import EuclideanGrid, Grid, ManhattanGrid
from .occupancy import Occupancy
