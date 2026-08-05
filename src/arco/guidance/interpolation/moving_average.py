"""MovingAverageInterpolator: endpoint-preserving polyline smoothing."""

from __future__ import annotations

from typing import Any, List

import numpy as np

from .base import Interpolator


class MovingAverageInterpolator(Interpolator):
    """Sliding-window moving-average smoothing of a waypoint polyline.

    Filters the high-frequency lateral wiggle that sampling planners
    (RRT*, SST) and grid staircases (A*) leave in their waypoint lists.
    Endpoints are preserved exactly; interior points are replaced by the
    window mean.  Repeated iterations increase the smoothing strength.

    The filter only ever pulls points toward the local chord, so lateral
    excursions shrink — but sharp real corners are also cut slightly.
    Callers tracking through obstacle fields should verify clearance
    after smoothing (see ``CityScene`` for a guarded application).
    """

    def __init__(self, *, iterations: int = 1, window: int = 3) -> None:
        """Initialize the moving-average smoother.

        Args:
            iterations: Number of smoothing passes (at least 1).
            window: Odd window size in waypoints (at least 3).

        Raises:
            ValueError: If *window* is even or smaller than 3.
        """
        if window < 3 or window % 2 == 0:
            raise ValueError("window must be an odd integer >= 3.")
        self.iterations = max(int(iterations), 1)
        self.window = int(window)

    def interpolate(self, path: List[Any]) -> List[Any]:
        """Smooth a discrete path with repeated moving-average passes.

        Args:
            path: A list of discrete waypoints (``(x, y)``-like).

        Returns:
            A list of ``(x, y)`` tuples with identical first and last
            points; inputs shorter than 3 points are returned unchanged.
        """
        if len(path) < 3:
            return list(path)
        pts = np.asarray([(float(p[0]), float(p[1])) for p in path])
        half = self.window // 2
        for _ in range(self.iterations):
            smoothed = pts.copy()
            for i in range(1, len(pts) - 1):
                lo = max(0, i - half)
                hi = min(len(pts), i + half + 1)
                smoothed[i] = pts[lo:hi].mean(axis=0)
            pts = smoothed
        return [(float(p[0]), float(p[1])) for p in pts]
