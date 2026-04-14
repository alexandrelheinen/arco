"""JointSpaceTracker: N-DOF C-space tracker with saturation and APF repulsion.

Generalises the per-robot tracker classes (PPPRobot, RRPRaceRobot) into a
single reusable component that works for any N-dimensional configuration
space.  The control law is a proportional position-to-velocity mapping
with independent per-axis velocity and acceleration saturation, plus an
optional Artificial Potential Field (APF) repulsion term that pushes the
configuration away from the nearest C-space obstacle.

Pipeline role
-------------
::

    planned / optimised trajectory
           │   (carrot position at current arc-length)
           ▼
    ┌──────────────────┐
    │ JointSpaceTracker│  step(target_q, dt) → new_q
    │  • P-control     │
    │  • vel sat       │
    │  • acc sat       │
    │  • APF repulsion │
    └──────────────────┘
           │
           ▼
       robot / body integrator
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from arco.mapping.occupancy import Occupancy


class JointSpaceTracker:
    """N-DOF proportional tracker with velocity/acceleration saturation and APF repulsion.

    Implements the reactive tracking layer of the planning pipeline for
    configuration-space agents (joint-space arms, Cartesian gantries,
    body-pose controllers).  Each call to :meth:`step` produces one
    integrated velocity step and updates the internal state.

    The control law is::

        desired_vel  = clip(k_p * (target - q), -max_vel, +max_vel)
        desired_vel += repulsion_velocity(q)          # APF correction
        desired_vel  = clip(desired_vel, -max_vel, +max_vel)
        dv           = clip(desired_vel - vel, -max_acc*dt, +max_acc*dt)
        vel          = clip(vel + dv, -max_vel, +max_vel)
        q           += vel * dt

    The APF repulsion velocity points away from the nearest C-space obstacle
    with magnitude ``gain × (1/d − 1/d_max)`` where ``d_max = 2 × clearance``
    is the influence radius.  When ``d = 0`` the distance is clamped to
    ``0.1 × clearance`` for numerical stability.

    Args:
        max_vel: Per-axis maximum velocity array (same units as the
            configuration space per second).  Accepts a scalar to apply
            the same limit to every axis.
        max_acc: Per-axis maximum acceleration (velocity units per second).
            Accepts a scalar.
        proportional_gain: Scalar P-gain mapping position error to desired
            velocity.  Defaults to ``2.0``.
        occupancy: Optional C-space occupancy map used for APF repulsion.
            Must expose :meth:`~arco.mapping.Occupancy.nearest_obstacle`.
            When ``None`` or when *repulsion_gain* is ``0.0``, repulsion
            is disabled.
        repulsion_gain: APF repulsion gain (velocity-units · world-units⁻¹).
            The maximum repulsion speed at ``d → 0`` scales as
            ``gain / (0.1 × clearance)``.  Typical range: ``0.1``–``2.0``.
            ``0.0`` disables repulsion.
    """

    def __init__(
        self,
        max_vel: float | np.ndarray,
        max_acc: float | np.ndarray,
        proportional_gain: float = 2.0,
        occupancy: Optional[Occupancy] = None,
        repulsion_gain: float = 0.0,
    ) -> None:
        """Initialize the JointSpaceTracker.

        Args:
            max_vel: Per-axis velocity limit (scalar or 1-D array).
            max_acc: Per-axis acceleration limit (scalar or 1-D array).
            proportional_gain: P-gain mapping error → desired velocity.
            occupancy: Optional C-space occupancy map for APF repulsion.
            repulsion_gain: APF gain; ``0.0`` disables repulsion.

        Raises:
            ValueError: If any element of *max_vel* or *max_acc* is
                not strictly positive.
        """
        self._max_vel = np.atleast_1d(np.asarray(max_vel, dtype=float))
        self._max_acc = np.atleast_1d(np.asarray(max_acc, dtype=float))
        if np.any(self._max_vel <= 0.0):
            raise ValueError(
                f"max_vel must be strictly positive; got {max_vel!r}."
            )
        if np.any(self._max_acc <= 0.0):
            raise ValueError(
                f"max_acc must be strictly positive; got {max_acc!r}."
            )
        self._k_p = float(proportional_gain)
        self._occ = occupancy
        self.repulsion_gain = float(repulsion_gain)
        # Internal state — reset via reset() before tracking.
        self.q: np.ndarray = np.zeros_like(self._max_vel)
        self.vel: np.ndarray = np.zeros_like(self._max_vel)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self, q0: np.ndarray) -> None:
        """Reset tracker state to initial configuration *q0*.

        Must be called before the first :meth:`step` when re-using the
        tracker for a new trajectory or after a replanning event.

        Args:
            q0: Initial configuration array (same length as the
                configured number of DOFs).
        """
        self.q = np.asarray(q0, dtype=float).copy()
        self.vel = np.zeros_like(self.q)

    # ------------------------------------------------------------------
    # Tracking step
    # ------------------------------------------------------------------

    def step(self, target_q: np.ndarray, dt: float) -> np.ndarray:
        """Run one tracker step toward *target_q* and return the new configuration.

        Computes the proportional velocity command, adds APF repulsion if
        enabled, saturates velocity and acceleration, integrates the state,
        and returns the updated configuration.

        Args:
            target_q: Carrot (target) configuration on the planned path.
                Must have the same length as the tracker DOFs.
            dt: Integration time step (seconds).  Must be positive.

        Returns:
            Updated configuration array after integration.
        """
        target = np.asarray(target_q, dtype=float)
        err = target - self.q

        # Proportional velocity command.
        desired_vel = np.clip(
            self._k_p * err, -self._max_vel, self._max_vel
        )

        # Add APF repulsion correction.
        repulsion = self._repulsion_velocity(self.q)
        desired_vel = desired_vel + repulsion

        # Re-clamp after repulsion to stay within velocity envelope.
        desired_vel = np.clip(desired_vel, -self._max_vel, self._max_vel)

        # Acceleration saturation.
        dv = np.clip(
            desired_vel - self.vel, -self._max_acc * dt, self._max_acc * dt
        )
        self.vel = np.clip(
            self.vel + dv, -self._max_vel, self._max_vel
        )

        # Euler integration.
        self.q = self.q + self.vel * dt
        return self.q.copy()

    # ------------------------------------------------------------------
    # APF repulsion
    # ------------------------------------------------------------------

    def _repulsion_velocity(self, q: np.ndarray) -> np.ndarray:
        """Compute APF obstacle-repulsion velocity correction in C-space.

        Returns a velocity vector pointing away from the nearest C-space
        obstacle when the configuration is within ``2 × clearance`` of it.
        The magnitude follows the standard APF formula::

            Δv = gain × (1/d − 1/d_max) × (q − nearest) / d

        where *d* is the distance to the nearest obstacle and
        *d_max* = 2 × clearance is the influence radius.

        Args:
            q: Current configuration as a numpy array.

        Returns:
            Velocity correction array (same shape as *q*); zero array when
            repulsion is disabled or the configuration is outside the
            influence radius.
        """
        if self._occ is None or self.repulsion_gain <= 0.0:
            return np.zeros_like(q)
        clearance: float = getattr(self._occ, "clearance", 0.0)
        if clearance <= 0.0 or not hasattr(self._occ, "nearest_obstacle"):
            return np.zeros_like(q)

        influence = 2.0 * clearance
        dist, nearest = self._occ.nearest_obstacle(q)  # type: ignore[attr-defined]
        if dist >= influence:
            return np.zeros_like(q)

        # Clamp distance for numerical stability.
        d_safe = max(dist, 0.1 * clearance)
        direction = (q - np.asarray(nearest, dtype=float)) / d_safe
        magnitude = self.repulsion_gain * (1.0 / d_safe - 1.0 / influence)
        return magnitude * direction
