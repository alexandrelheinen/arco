"""
Mapping module for spatial representations.

This module provides base classes for grid and occupancy maps, following the ARCO architecture guidelines.
"""

from .graph import Graph
from .grid.base import Grid
from .grid.euclidean import EuclideanGrid
from .grid.manhattan import ManhattanGrid
from .occupancy import Occupancy
from .oriented_graph import OrientedGraph
from .weighted_graph import WeightedGraph
