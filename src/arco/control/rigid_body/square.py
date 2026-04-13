"""Uniform-density square rigid body."""

from __future__ import annotations

import math

import numpy as np

from arco.control.rigid_body.base import RigidBody


class SquareBody(RigidBody):
    """Uniform-density square rigid body.

    Inertia: I = mass * side_length² / 6  (thin uniform square plate).
    Bounding radius: side_length * √2 / 2.

    Args:
        mass: Body mass in kg. Must be positive.
        side_length: Length of one side in metres. Must be positive.
        x: Initial x-position in metres.
        y: Initial y-position in metres.
        psi: Initial heading angle in radians.

    Raises:
        ValueError: If mass or side_length is not strictly positive.
    """

    def __init__(
        self,
        mass: float,
        side_length: float,
        x: float = 0.0,
        y: float = 0.0,
        psi: float = 0.0,
    ) -> None:
        super().__init__(mass=mass, x=x, y=y, psi=psi)
        if side_length <= 0.0:
            raise ValueError(
                f"side_length must be positive, got {side_length!r}."
            )
        self._side_length = float(side_length)

    @property
    def side_length(self) -> float:
        """Length of one side in metres."""
        return self._side_length

    @property
    def inertia(self) -> float:
        """Rotational inertia: I = mass * side_length² / 6."""
        return self._mass * self._side_length**2 / 6.0

    @property
    def bounding_radius(self) -> float:
        """Bounding circle radius: side_length * √2 / 2."""
        return self._side_length * math.sqrt(2.0) / 2.0

    def corners(self) -> np.ndarray:
        """Return the 4 corners in world frame as a (4, 2) array.

        Returns:
            Array of shape (4, 2) with corner positions (x, y).
        """
        a = self._side_length / 2.0
        cos_psi = math.cos(self._pose[2])
        sin_psi = math.sin(self._pose[2])
        corners_body = np.array(
            [(-a, -a), (a, -a), (a, a), (-a, a)], dtype=float
        )
        cx, cy = self._pose[0], self._pose[1]
        result = np.empty((4, 2), dtype=float)
        for i, (bx, by) in enumerate(corners_body):
            result[i, 0] = cx + cos_psi * bx - sin_psi * by
            result[i, 1] = cy + sin_psi * bx + cos_psi * by
        return result
