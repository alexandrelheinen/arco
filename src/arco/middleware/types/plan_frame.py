"""PlanFrame: typed dataclass for planning-layer bus messages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class PlanFrame:
    """A single snapshot produced by the planning layer.

    Published to the shared in-memory bus once a path has been found by
    a sampling-based or graph-search planner.  Consumers (guidance layer,
    visualizers) receive this frame to perform trajectory optimization or
    rendering.

    Attributes:
        timestamp: Seconds since pipeline start, from
            ``time.monotonic()``.
        waypoints: Ordered list of waypoints forming the raw planned
            path; each entry is a coordinate sequence ``[x, y, …]`` in
            configuration-space units.
        planner: Human-readable name of the algorithm that produced
            this path (e.g. ``"RRT*"`` or ``"SST"``).
    """

    timestamp: float
    waypoints: List[List[float]] = field(default_factory=list)
    planner: str = ""
