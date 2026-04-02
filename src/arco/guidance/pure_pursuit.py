"""PurePursuitController: pure pursuit path tracking controller."""

from __future__ import annotations

import math
from typing import Sequence

from .controller import Controller


class PurePursuitController(Controller):
    """Pure pursuit controller for 2-D path tracking.

    Computes a turn-rate command that steers a unicycle vehicle toward a
    lookahead point located a fixed arc length ahead along the reference path.
    Cross-track error and heading error are updated on every :meth:`track`
    call and exposed as read-only attributes for logging.

    The standard pure pursuit turn-rate law is::

        ω = 2 · v · sin(α) / L_d

    where *α* is the bearing from the vehicle heading to the lookahead
    direction in the vehicle frame and *L_d* is ``lookahead_distance``.

    Attributes:
        lookahead_distance: Arc length used to locate the lookahead point (m).
        cross_track_error: Signed perpendicular distance from vehicle to the
            nearest path segment (positive = vehicle is left of path).
        heading_error: Difference between vehicle heading and path tangent
            at the nearest segment, wrapped to ``(−π, π]`` (radians).
    """

    def __init__(self, lookahead_distance: float = 1.0) -> None:
        """Initialize PurePursuitController.

        Args:
            lookahead_distance: Arc length ahead on the path used to compute
                the lookahead point (metres).
        """
        self.lookahead_distance = lookahead_distance
        self.cross_track_error: float = 0.0
        self.heading_error: float = 0.0

    def track(
        self,
        pose: tuple[float, float, float],
        path: Sequence[tuple[float, float]],
        speed: float = 1.0,
    ) -> tuple[float, float]:
        """Compute pure pursuit speed and turn-rate commands.

        Finds the lookahead point on *path* that is approximately
        ``lookahead_distance`` metres ahead of the vehicle and computes the
        instantaneous turn rate that steers the vehicle toward it.  Also
        updates :attr:`cross_track_error` and :attr:`heading_error`.

        Args:
            pose: Current vehicle pose ``(x, y, heading)`` in world frame.
            path: Ordered sequence of ``(x, y)`` waypoints.
            speed: Desired forward speed (m/s) passed through as-is.

        Returns:
            Tuple ``(speed_cmd, turn_rate_cmd)`` where ``speed_cmd`` equals
            *speed* and ``turn_rate_cmd`` is in rad/s.
        """
        if len(path) < 2:
            return speed, 0.0

        x, y, theta = pose

        # --- Find closest waypoint index ---
        min_dist = math.inf
        closest_idx = 0
        for i, (wx, wy) in enumerate(path):
            dist = math.hypot(wx - x, wy - y)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # --- Cross-track and heading errors at closest segment ---
        if closest_idx < len(path) - 1:
            sx = path[closest_idx + 1][0] - path[closest_idx][0]
            sy = path[closest_idx + 1][1] - path[closest_idx][1]
        else:
            sx = path[closest_idx][0] - path[closest_idx - 1][0]
            sy = path[closest_idx][1] - path[closest_idx - 1][1]

        seg_len = math.hypot(sx, sy)
        if seg_len > 1e-9:
            # Left-pointing unit normal of the path segment
            nx = -sy / seg_len
            ny = sx / seg_len
            self.cross_track_error = nx * (x - path[closest_idx][0]) + ny * (
                y - path[closest_idx][1]
            )
            self.heading_error = _wrap_angle(theta - math.atan2(sy, sx))
        else:
            self.cross_track_error = 0.0
            self.heading_error = 0.0

        # --- Find lookahead point ---
        lookahead = _find_lookahead(
            x, y, path, closest_idx, self.lookahead_distance
        )

        # --- Pure pursuit turn-rate law ---
        lx, ly = lookahead
        dx = lx - x
        dy = ly - y
        # Transform lookahead vector to vehicle frame
        dx_v = math.cos(theta) * dx + math.sin(theta) * dy
        dy_v = -math.sin(theta) * dx + math.cos(theta) * dy
        alpha = math.atan2(dy_v, dx_v)
        turn_rate_cmd = 2.0 * speed * math.sin(alpha) / self.lookahead_distance

        return speed, turn_rate_cmd

    def control(self, state: float, reference: float) -> float:
        """Compute a proportional steering command from a scalar error.

        This method satisfies the :class:`Controller` interface for simple
        scalar inputs.  For full 2-D path tracking use :meth:`track`.

        Args:
            state: The current state value.
            reference: The reference/target value.

        Returns:
            Proportional command ``reference − state`` as a float.
        """
        return reference - state


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _wrap_angle(angle: float) -> float:
    """Wrap *angle* to the interval ``(−π, π]``.

    Args:
        angle: Angle in radians.

    Returns:
        Equivalent angle in ``(−π, π]``.
    """
    return math.atan2(math.sin(angle), math.cos(angle))


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
    ahead.  Returns the last waypoint when no segment intersection is found
    at the requested distance.

    Args:
        x: Vehicle x position.
        y: Vehicle y position.
        path: Ordered sequence of ``(x, y)`` waypoints.
        start_idx: Index of the closest waypoint on the path.
        lookahead: Desired lookahead distance (metres).

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
    return path[-1]


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
        cx: Circle centre x.
        cy: Circle centre y.
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
