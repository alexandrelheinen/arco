"""PathTracker: duck-typed contract for geometric path trackers."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class PathTracker(Protocol):
    """Geometric path tracker used by :class:`~arco.control.tracking.TrackingLoop`.

    Matches :class:`~arco.control.pure_pursuit.PurePursuitController.track`
    plus the error/curvature attributes the loop reads after each call.
    """

    cross_track_error: float
    heading_error: float
    curvature: float

    def track(
        self,
        pose: tuple[float, float, float],
        path: Sequence[tuple[float, float]],
        speed: float = 1.0,
    ) -> tuple[float, float]:
        """Return ``(speed_cmd, turn_rate_cmd)`` for the current pose."""
