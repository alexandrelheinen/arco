"""MappingFrame: typed dataclass for mapping-layer bus messages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class MappingFrame:
    """A single snapshot produced by the mapping layer.

    Published to the shared in-memory bus once the occupancy map has been
    built from the configuration file.  Consumers (planners, visualizers)
    receive this frame and can derive obstacle queries from it.

    Attributes:
        timestamp: Seconds since pipeline start, from
            ``time.monotonic()``.
        obstacle_points: List of obstacle sample points; each entry is
            a coordinate sequence ``[x, y, …]`` in configuration-space
            units.
        bounds: Axis-aligned bounding box ``[x_min, y_min, x_max,
            y_max]`` of the planning domain.
        clearance: Minimum obstacle-free margin used during map
            construction.
    """

    timestamp: float
    obstacle_points: List[List[float]] = field(default_factory=list)
    bounds: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 1.0])
    clearance: float = 0.0
