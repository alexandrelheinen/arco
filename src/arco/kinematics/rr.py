"""RRRobot: two-link planar revolute-revolute arm kinematics.

The arm is mounted at the world origin and operates in the XY plane.
Joint 1 rotates the first link of length *l1* around the Z axis; joint 2
rotates the second link of length *l2* relative to the first link.

Coordinate convention (z-up, z not used):
    - Origin (0, 0) is the base of joint 1.
    - Joint 2 position: ``(l1 cos q1, l1 sin q1)``.
    - End-effector position: joint2 + ``(l2 cos(q1+q2), l2 sin(q1+q2))``.

Example::

    robot = RRRobot(l1=1.0, l2=0.8)
    x, y = robot.forward_kinematics(0.0, 0.0)   # (1.8, 0.0)
    solutions = robot.inverse_kinematics(1.4, 0.5)
"""

from __future__ import annotations

import math


class RRRobot:
    """Two-link planar revolute-revolute robot arm.

    Args:
        l1: Length of the first link in metres.  Must be positive.
        l2: Length of the second link in metres.  Must be positive.
            The second link is typically shorter than the first (e.g.
            ``l2 = 0.8 * l1``) to model a SCARA-style pick-and-place arm.

    Raises:
        ValueError: If *l1* or *l2* is not strictly positive.
    """

    def __init__(self, l1: float = 1.0, l2: float = 0.8) -> None:
        if l1 <= 0:
            raise ValueError(f"l1 must be positive, got {l1!r}.")
        if l2 <= 0:
            raise ValueError(f"l2 must be positive, got {l2!r}.")
        self._l1 = float(l1)
        self._l2 = float(l2)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def l1(self) -> float:
        """Length of the first link in metres."""
        return self._l1

    @property
    def l2(self) -> float:
        """Length of the second link in metres."""
        return self._l2

    # ------------------------------------------------------------------
    # Forward kinematics
    # ------------------------------------------------------------------

    def forward_kinematics(self, q1: float, q2: float) -> tuple[float, float]:
        """Compute end-effector position from joint angles.

        Args:
            q1: First joint angle in radians (rotation of link 1 around
                the world Z axis).
            q2: Second joint angle in radians (rotation of link 2
                relative to link 1).

        Returns:
            ``(x, y)`` Cartesian position of the end-effector in metres.
        """
        j2x = self._l1 * math.cos(q1)
        j2y = self._l1 * math.sin(q1)
        eex = j2x + self._l2 * math.cos(q1 + q2)
        eey = j2y + self._l2 * math.sin(q1 + q2)
        return float(eex), float(eey)

    # ------------------------------------------------------------------
    # Inverse kinematics
    # ------------------------------------------------------------------

    def inverse_kinematics(
        self,
        x: float,
        y: float,
        eps: float = 1e-9,
    ) -> list[tuple[float, float]]:
        """Compute joint angles that place the end-effector at *(x, y)*.

        Uses the law of cosines.  Returns up to two solutions:

        * *elbow-down* (positive q2) and *elbow-up* (negative q2).

        Points outside the reachable annulus
        ``[|l1 - l2|, l1 + l2]`` return an empty list.

        Args:
            x: Desired end-effector x coordinate in metres.
            y: Desired end-effector y coordinate in metres.
            eps: Numerical tolerance for workspace boundary checks.

        Returns:
            List of ``(q1, q2)`` tuples (radians).  The list has 0, 1,
            or 2 elements.
        """
        r_sq = x * x + y * y
        r = math.sqrt(r_sq)
        r_min, r_max = self.workspace_annulus()
        if r > r_max + eps or r < r_min - eps:
            return []

        l1, l2 = self._l1, self._l2
        # cos(q2) from the law of cosines:  r² = l1² + l2² + 2*l1*l2*cos(q2)
        cos_q2 = (r_sq - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
        cos_q2 = max(-1.0, min(1.0, cos_q2))  # clamp numerical noise

        solutions: list[tuple[float, float]] = []
        for sign in (1.0, -1.0):
            q2 = sign * math.acos(cos_q2)
            # q1 from atan2 of the target direction minus elbow offset
            k1 = l1 + l2 * math.cos(q2)
            k2 = l2 * math.sin(q2)
            q1 = math.atan2(y, x) - math.atan2(k2, k1)
            solutions.append((float(q1), float(q2)))
        return solutions

    # ------------------------------------------------------------------
    # Link segment geometry
    # ------------------------------------------------------------------

    def link_segments(
        self, q1: float, q2: float
    ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        """Return the three key points of the arm geometry.

        Args:
            q1: First joint angle in radians.
            q2: Second joint angle in radians.

        Returns:
            A tuple ``(origin, joint2, end_effector)`` where each element
            is an ``(x, y)`` pair in metres.
        """
        origin = (0.0, 0.0)
        j2x = self._l1 * math.cos(q1)
        j2y = self._l1 * math.sin(q1)
        eex, eey = self.forward_kinematics(q1, q2)
        return origin, (float(j2x), float(j2y)), (float(eex), float(eey))

    # ------------------------------------------------------------------
    # Workspace helpers
    # ------------------------------------------------------------------

    def workspace_radius(self) -> float:
        """Return the maximum reach (outer radius) of the arm.

        Returns:
            ``l1 + l2`` in metres.
        """
        return self._l1 + self._l2

    def workspace_annulus(self) -> tuple[float, float]:
        """Return the inner and outer radii of the reachable annulus.

        The inner radius is ``|l1 - l2|``.  When the links have equal
        length the inner radius is zero (the arm can reach the origin).

        Returns:
            ``(r_min, r_max)`` in metres.
        """
        return abs(self._l1 - self._l2), self._l1 + self._l2
