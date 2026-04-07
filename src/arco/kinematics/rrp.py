"""RRPRobot: two-link planar revolute-revolute arm with a vertical prismatic joint.

The arm is a SCARA-like robot.  Joints 1 and 2 are revolute (identical to
:class:`~arco.kinematics.rr.RRRobot`) and operate in the XY plane at the
current Z height.  Joint 3 is a prismatic joint that moves the entire arm
assembly up and down along the world Z axis.

Coordinate convention (z-up):
    - Base origin ``(0, 0, z)`` where *z* is the prismatic joint value.
    - Joint-2 position: ``(l1 cos q1,  l1 sin q1,  z)``.
    - End-effector:    ``(l1 cos q1 + l2 cos(q1+q2),  l1 sin q1 + l2 sin(q1+q2),  z)``.

Example::

    robot = RRPRobot(l1=1.0, l2=0.8, z_min=0.0, z_max=4.0)
    x, y, z = robot.forward_kinematics(0.0, 0.0, 2.0)   # (1.8, 0.0, 2.0)
"""

from __future__ import annotations

import math


class RRPRobot:
    """Two-link planar RR arm with a vertical prismatic joint (SCARA-like).

    The XY kinematics are identical to :class:`~arco.kinematics.rr.RRRobot`.
    The prismatic joint *z* translates the whole arm assembly vertically.

    Args:
        l1: Length of the first revolute link in metres.  Must be positive.
        l2: Length of the second revolute link in metres.  Must be positive.
        z_min: Minimum height of the prismatic joint in metres.
        z_max: Maximum height of the prismatic joint in metres.
            Must be strictly greater than *z_min*.

    Raises:
        ValueError: If *l1* or *l2* is not strictly positive, or if
            *z_max* ≤ *z_min*.
    """

    def __init__(
        self,
        l1: float = 1.0,
        l2: float = 0.8,
        z_min: float = 0.0,
        z_max: float = 4.0,
    ) -> None:
        if l1 <= 0:
            raise ValueError(f"l1 must be positive, got {l1!r}.")
        if l2 <= 0:
            raise ValueError(f"l2 must be positive, got {l2!r}.")
        if z_max <= z_min:
            raise ValueError(
                f"z_max ({z_max!r}) must be greater than z_min ({z_min!r})."
            )
        self._l1 = float(l1)
        self._l2 = float(l2)
        self._z_min = float(z_min)
        self._z_max = float(z_max)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def l1(self) -> float:
        """Length of the first revolute link in metres."""
        return self._l1

    @property
    def l2(self) -> float:
        """Length of the second revolute link in metres."""
        return self._l2

    @property
    def z_min(self) -> float:
        """Minimum prismatic joint height in metres."""
        return self._z_min

    @property
    def z_max(self) -> float:
        """Maximum prismatic joint height in metres."""
        return self._z_max

    # ------------------------------------------------------------------
    # Forward kinematics
    # ------------------------------------------------------------------

    def forward_kinematics(
        self, q1: float, q2: float, z: float
    ) -> tuple[float, float, float]:
        """Compute end-effector 3-D position from joint values.

        Args:
            q1: First revolute joint angle in radians.
            q2: Second revolute joint angle relative to link 1 in radians.
            z: Prismatic joint height in metres.

        Returns:
            ``(x, y, z)`` Cartesian position of the end-effector in metres.
        """
        j2x = self._l1 * math.cos(q1)
        j2y = self._l1 * math.sin(q1)
        eex = j2x + self._l2 * math.cos(q1 + q2)
        eey = j2y + self._l2 * math.sin(q1 + q2)
        return float(eex), float(eey), float(z)

    # ------------------------------------------------------------------
    # Inverse kinematics (XY only — z is passed through)
    # ------------------------------------------------------------------

    def inverse_kinematics_xy(
        self,
        x: float,
        y: float,
        eps: float = 1e-9,
    ) -> list[tuple[float, float]]:
        """Compute revolute joint angles that place the end-effector at *(x, y)*.

        The prismatic joint value does not affect the XY kinematics and is
        not involved in this calculation.

        Returns up to two solutions:

        * *elbow-down* (positive q2) and *elbow-up* (negative q2).

        Points outside the reachable annulus
        ``[|l1 - l2|, l1 + l2]`` return an empty list.

        Args:
            x: Desired end-effector x coordinate in metres.
            y: Desired end-effector y coordinate in metres.
            eps: Numerical tolerance for workspace boundary checks.

        Returns:
            List of ``(q1, q2)`` tuples (radians).
        """
        r_sq = x * x + y * y
        r = math.sqrt(r_sq)
        r_min, r_max = self.workspace_annulus()
        if r > r_max + eps or r < r_min - eps:
            return []

        l1, l2 = self._l1, self._l2
        cos_q2 = (r_sq - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
        cos_q2 = max(-1.0, min(1.0, cos_q2))

        solutions: list[tuple[float, float]] = []
        for sign in (1.0, -1.0):
            q2 = sign * math.acos(cos_q2)
            k1 = l1 + l2 * math.cos(q2)
            k2 = l2 * math.sin(q2)
            q1 = math.atan2(y, x) - math.atan2(k2, k1)
            solutions.append((float(q1), float(q2)))
        return solutions

    # ------------------------------------------------------------------
    # Link segment geometry
    # ------------------------------------------------------------------

    def link_segments(self, q1: float, q2: float, z: float) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]:
        """Return the three key 3-D points of the arm geometry.

        Args:
            q1: First revolute joint angle in radians.
            q2: Second revolute joint angle in radians.
            z: Prismatic joint height in metres.

        Returns:
            A tuple ``(origin, joint2, end_effector)`` where each element
            is an ``(x, y, z)`` triple in metres.
        """
        j2x = self._l1 * math.cos(q1)
        j2y = self._l1 * math.sin(q1)
        eex, eey, ez = self.forward_kinematics(q1, q2, z)
        origin = (0.0, 0.0, float(z))
        joint2 = (float(j2x), float(j2y), float(z))
        ee = (float(eex), float(eey), float(ez))
        return origin, joint2, ee

    # ------------------------------------------------------------------
    # Workspace helpers
    # ------------------------------------------------------------------

    def workspace_radius(self) -> float:
        """Return the maximum horizontal reach (outer radius) of the arm.

        Returns:
            ``l1 + l2`` in metres.
        """
        return self._l1 + self._l2

    def workspace_annulus(self) -> tuple[float, float]:
        """Return the inner and outer radii of the reachable horizontal annulus.

        Returns:
            ``(r_min, r_max)`` in metres, where ``r_min = |l1 - l2|``.
        """
        return abs(self._l1 - self._l2), self._l1 + self._l2
