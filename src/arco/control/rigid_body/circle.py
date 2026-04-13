"""Uniform-density circular rigid body."""

from __future__ import annotations

from arco.control.rigid_body.base import RigidBody


class CircleBody(RigidBody):
    """Uniform-density circular rigid body.

    Inertia: I = mass * radius² / 2  (thin uniform disk).
    Bounding radius: radius.

    Args:
        mass: Body mass in kg. Must be positive.
        radius: Circle radius in metres. Must be positive.
        x: Initial x-position in metres.
        y: Initial y-position in metres.
        psi: Initial heading angle in radians.

    Raises:
        ValueError: If mass or radius is not strictly positive.
    """

    def __init__(
        self,
        mass: float,
        radius: float,
        x: float = 0.0,
        y: float = 0.0,
        psi: float = 0.0,
    ) -> None:
        super().__init__(mass=mass, x=x, y=y, psi=psi)
        if radius <= 0.0:
            raise ValueError(f"radius must be positive, got {radius!r}.")
        self._radius = float(radius)

    @property
    def radius(self) -> float:
        """Circle radius in metres."""
        return self._radius

    @property
    def inertia(self) -> float:
        """Rotational inertia: I = mass * radius² / 2."""
        return self._mass * self._radius**2 / 2.0

    @property
    def bounding_radius(self) -> float:
        """Bounding circle radius equals the body radius."""
        return self._radius
