"""Pure pursuit path tracking, re-exported from the compiled extension.

The implementation is ``PurePursuitTracker`` in the ``arco-control``
crate, registered back under its Python spelling by the binding layer and
reaching callers through :mod:`arco._arco`.

The two private helpers below stay in Python. The crate computes the
lookahead point inside the tracker rather than exposing it, so there is no
compiled counterpart to re-export, and the regression tests that pin the
off-track fallback call :func:`_find_lookahead` directly.
"""

from __future__ import annotations

import math
from typing import Sequence

from arco._arco import PurePursuitController

__all__ = ["PurePursuitController"]


def _find_lookahead(
    x: float,
    y: float,
    path: Sequence[tuple[float, float]],
    start_idx: int,
    lookahead: float,
) -> tuple[float, float]:
    """Return the lookahead point on *path* at distance *lookahead* from vehicle.

    Searches forward along the path starting from the segment that *ends* at
    ``path[start_idx]`` (i.e. from ``max(0, start_idx - 1)``).  Including
    the preceding segment ensures the lookahead is found correctly when the
    vehicle is between two waypoints and the closest waypoint is the one
    ahead.

    When no segment intersection is found (e.g. the vehicle drifted far
    off-track so the lookahead circle does not reach the path), the function
    returns the **next forward waypoint** ``path[start_idx + 1]`` rather than
    ``path[-1]`` (the goal).  This ensures the vehicle is steered back toward
    the correct path segment instead of jumping straight to the goal.

    Args:
        x: Vehicle x position.
        y: Vehicle y position.
        path: Ordered sequence of ``(x, y)`` waypoints.
        start_idx: Index of the closest waypoint on the path.
        lookahead: Desired lookahead distance (meters).

    Returns:
        ``(x, y)`` coordinates of the lookahead point.
    """
    for i in range(max(0, start_idx - 1), len(path) - 1):
        p0x, p0y = path[i]
        p1x, p1y = path[i + 1]
        # Only inspect segments whose far end is at least *lookahead* away
        if math.hypot(p1x - x, p1y - y) >= lookahead:
            pt = _circle_segment_intersection(
                x, y, lookahead, p0x, p0y, p1x, p1y
            )
            if pt is not None:
                return pt
    # Fallback: steer toward the next forward waypoint rather than the goal.
    # This handles the case where the vehicle is so far off-track that the
    # lookahead circle cannot intersect any path segment.  Returning path[-1]
    # here would cause the vehicle to bypass all remaining waypoints.
    next_idx = min(start_idx + 1, len(path) - 1)
    next_pt = path[next_idx]
    return (float(next_pt[0]), float(next_pt[1]))


def _circle_segment_intersection(
    cx: float,
    cy: float,
    r: float,
    p0x: float,
    p0y: float,
    p1x: float,
    p1y: float,
) -> tuple[float, float] | None:
    """Intersect a circle with a line segment and return the farther intersection.

    Solves the quadratic that arises from substituting the parametric segment
    equation into the circle equation, then returns the parameter value closest
    to the segment end (i.e. the intersection farthest along the path).

    Args:
        cx: Circle center x.
        cy: Circle center y.
        r: Circle radius.
        p0x: Segment start x.
        p0y: Segment start y.
        p1x: Segment end x.
        p1y: Segment end y.

    Returns:
        Intersection point ``(x, y)`` closest to the segment end, or ``None``
        if there is no intersection within the segment.
    """
    dx = p1x - p0x
    dy = p1y - p0y
    fx = p0x - cx
    fy = p0y - cy

    a = dx * dx + dy * dy
    if a < 1e-12:
        return None
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0:
        return None

    sqrt_disc = math.sqrt(discriminant)
    t2 = (-b + sqrt_disc) / (2.0 * a)
    t1 = (-b - sqrt_disc) / (2.0 * a)

    # Prefer t2 (farther along segment, i.e. closer to the end)
    for t in (t2, t1):
        if 0.0 <= t <= 1.0:
            return (p0x + t * dx, p0y + t * dy)

    return None
