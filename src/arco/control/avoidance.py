"""ArtificialPotentialField: reactive APF turn-rate avoidance strategy."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from arco.mapping.occupancy import Occupancy


class ArtificialPotentialField:
    """Artificial Potential Field turn-rate bias for obstacle avoidance.

    Implements the reactive APF correction historically inlined in
    :class:`~arco.control.tracking.TrackingLoop`.  When the vehicle is
    within ``2 × clearance`` of the nearest obstacle, returns an additive
    turn-rate bias that steers away from the obstacle.  The magnitude is
    proportional to ``repulsion_gain × (1/d − 1/d_max)`` where *d* is the
    obstacle distance and *d_max* = 2 × clearance.

    Callables of this class satisfy
    :class:`~arco.protocols.avoidance.AvoidanceStrategy`.

    A no-op / disabled instance is obtained by constructing with
    ``occupancy=None`` and/or ``repulsion_gain <= 0``; ``__call__`` then
    always returns ``0.0``.

    Attributes:
        repulsion_gain: Obstacle-repulsion turn-rate gain (rad/m).  A
            value of ``0.0`` disables repulsion.
    """

    def __init__(
        self,
        occupancy: Optional["Occupancy"] = None,
        repulsion_gain: float = 0.0,
    ) -> None:
        """Initialize ArtificialPotentialField.

        Args:
            occupancy: Optional occupancy map used to query the nearest
                obstacle.  When ``None`` (default), every call returns
                ``0.0``.
            repulsion_gain: Obstacle-repulsion turn-rate gain (rad/m).
                Non-positive values disable repulsion (no-op path).
                Typical range: ``0.5``–``3.0``.
        """
        self._occupancy = occupancy
        self.repulsion_gain = float(repulsion_gain)

    def __call__(self, x: float, y: float, theta: float) -> float:
        """Return an APF obstacle-repulsion turn-rate correction.

        Returns an additive turn-rate (rad/s) that steers the vehicle away
        from the nearest obstacle when it is within ``2 × clearance``
        meters.  The magnitude follows the standard APF formula::

            Δω = −gain × (1/d − 1/d_max) × sign(lateral)

        where *d* is the distance to the nearest obstacle, *d_max* is the
        influence radius (2 × clearance), and *lateral* is the signed
        lateral displacement of the obstacle from the vehicle's heading
        (positive = obstacle is to the vehicle's left).

        The sign convention steers *away* from the obstacle:
        obstacle to the left  → negative Δω (turn right);
        obstacle to the right → positive Δω (turn left).

        Args:
            x: Vehicle x-position in world frame (m).
            y: Vehicle y-position in world frame (m).
            theta: Vehicle heading in radians.

        Returns:
            Turn-rate correction in rad/s; ``0.0`` when disabled, outside
            the influence radius, or when no occupancy map is configured.
        """
        if self._occupancy is None or self.repulsion_gain <= 0.0:
            return 0.0
        clearance: float = getattr(self._occupancy, "clearance", 0.0)
        if clearance <= 0.0 or not hasattr(
            self._occupancy, "nearest_obstacle"
        ):
            return 0.0

        influence_radius = 2.0 * clearance
        pt = np.array([x, y], dtype=float)
        dist, nearest = self._occupancy.nearest_obstacle(pt)  # type: ignore[attr-defined]
        if dist >= influence_radius or dist < 1e-6:
            return 0.0

        # Signed lateral displacement of the obstacle from the vehicle axis.
        # Vehicle lateral direction (pointing LEFT of heading) is
        # (−sin θ, cos θ).
        dx = float(nearest[0]) - x
        dy = float(nearest[1]) - y
        lateral = dx * (-math.sin(theta)) + dy * math.cos(theta)

        # APF magnitude: (1/d − 1/d_max) with distance clamped for stability.
        magnitude = self.repulsion_gain * (
            1.0 / max(dist, 0.1 * clearance) - 1.0 / influence_radius
        )

        # Steer away: obstacle to the left (lateral > 0) → turn right (Δω < 0)
        lateral_sign = math.copysign(1.0, lateral)
        return -magnitude * lateral_sign
