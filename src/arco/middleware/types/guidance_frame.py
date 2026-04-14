"""GuidanceFrame: typed dataclass for guidance-layer bus messages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class GuidanceFrame:
    """A single snapshot produced by the guidance layer.

    Published to the shared in-memory bus once trajectory optimization
    has refined the raw plan into a time-parameterized trajectory.
    Consumers (frontends such as arcoex or arcosim) subscribe to this
    frame type and render or record the trajectory at their own pace.

    Attributes:
        timestamp: Seconds since pipeline start, from
            ``time.monotonic()``.
        trajectory: Ordered list of optimized trajectory points; each
            entry is a coordinate sequence ``[x, y, …]`` in
            configuration-space units.
        durations: Time duration (seconds) associated with each
            segment between consecutive trajectory points.  Length
            equals ``len(trajectory) - 1`` for a valid trajectory.
    """

    timestamp: float
    trajectory: List[List[float]] = field(default_factory=list)
    durations: List[float] = field(default_factory=list)
