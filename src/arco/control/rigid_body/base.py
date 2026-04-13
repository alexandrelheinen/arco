"""Abstract 2-D rigid body base class."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class RigidBody(ABC):
    """Abstract 2-D rigid body.

    State vector: pose (x, y, psi) and velocity (vx, vy, omega).

    Args:
        mass: Body mass in kg. Must be positive.
        x: Initial x-position in metres.
        y: Initial y-position in metres.
        psi: Initial heading angle in radians.

    Raises:
        ValueError: If mass is not strictly positive.
    """

    def __init__(
        self,
        mass: float,
        x: float = 0.0,
        y: float = 0.0,
        psi: float = 0.0,
    ) -> None:
        if mass <= 0.0:
            raise ValueError(f"mass must be positive, got {mass!r}.")
        self._mass = float(mass)
        self._pose = np.array([x, y, psi], dtype=float)
        self._velocity = np.zeros(3, dtype=float)
        self._accumulated_force = np.zeros(2, dtype=float)
        self._accumulated_torque = 0.0

    @property
    @abstractmethod
    def inertia(self) -> float:
        """Rotational inertia about the body's centre of mass (kg·m²)."""

    @property
    @abstractmethod
    def bounding_radius(self) -> float:
        """Bounding circle radius for the body (metres)."""

    @property
    def mass(self) -> float:
        """Body mass in kg."""
        return self._mass

    @property
    def pose(self) -> np.ndarray:
        """Current pose as [x, y, psi]."""
        return self._pose.copy()

    @property
    def velocity(self) -> np.ndarray:
        """Current velocity as [vx, vy, omega]."""
        return self._velocity.copy()

    def apply_wrench(
        self,
        fx: float,
        fy: float,
        torque: float,
    ) -> None:
        """Accumulate a wrench to be applied at the next step.

        Wrenches are cleared after each call to :meth:`step`.

        Args:
            fx: Force in world x-direction (N).
            fy: Force in world y-direction (N).
            torque: Torque about z-axis (N·m).
        """
        self._accumulated_force[0] += fx
        self._accumulated_force[1] += fy
        self._accumulated_torque += torque

    def step(self, dt: float) -> None:
        """Integrate dynamics forward by dt seconds using Euler integration.

        Applies accumulated wrenches then clears them.

        Args:
            dt: Time step in seconds. Must be positive.
        """
        ax = self._accumulated_force[0] / self._mass
        ay = self._accumulated_force[1] / self._mass
        alpha = self._accumulated_torque / self.inertia

        self._velocity[0] += ax * dt
        self._velocity[1] += ay * dt
        self._velocity[2] += alpha * dt

        self._pose[0] += self._velocity[0] * dt
        self._pose[1] += self._velocity[1] * dt
        self._pose[2] += self._velocity[2] * dt

        self._accumulated_force[:] = 0.0
        self._accumulated_torque = 0.0

    def reset(
        self,
        x: float = 0.0,
        y: float = 0.0,
        psi: float = 0.0,
    ) -> None:
        """Reset pose to (x, y, psi) and zero velocity.

        Args:
            x: New x-position.
            y: New y-position.
            psi: New heading angle.
        """
        self._pose[:] = [x, y, psi]
        self._velocity[:] = 0.0
        self._accumulated_force[:] = 0.0
        self._accumulated_torque = 0.0
