"""TrackingLoop: local control loop for route following with bounded dynamics."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from arco.guidance.vehicle import DubinsVehicle
    from arco.mapping.occupancy import Occupancy

import numpy as np

from .pure_pursuit import PurePursuitController


class TrackingLoop:
    """Local tracking loop combining a vehicle model and a path controller.

    Closes the feedback loop between a :class:`~arco.guidance.vehicle.DubinsVehicle`
    kinematic model and a :class:`PurePursuitController`.  Each call to
    :meth:`step` issues pure pursuit commands to the vehicle and records
    cross-track error, heading error, pose, speed, and turn rate for later
    analysis.

    When an *occupancy* map and a positive *repulsion_gain* are provided, a
    reactive obstacle-repulsion correction is blended into the turn-rate
    command at every step.  The correction is an Artificial Potential Field
    (APF) lateral force: when the vehicle is within ``2 × clearance`` of the
    nearest obstacle, a turn-rate bias is added that steers the vehicle away
    from the obstacle.  The bias magnitude is proportional to
    ``repulsion_gain × (1/d − 1/d_max)`` where *d* is the obstacle distance
    and *d_max* = 2 × clearance is the influence radius.  This prevents the
    vehicle from crashing into obstacles even when the planned trajectory
    skirts obstacle boundaries.

    Attributes:
        vehicle: Kinematic vehicle model.
        controller: Pure pursuit path-tracking controller.
        cruise_speed: Desired forward speed passed to the controller (m/s).
        curvature_gain: Curvature-to-speed scaling factor (m).  Speed is
            modulated as ``v = cruise_speed / (1 + curvature_gain * |κ|)``
            where *κ* is the pure pursuit curvature from the previous step.
            A value of ``0.0`` (default) disables modulation.
        repulsion_gain: Obstacle-repulsion turn-rate gain (rad/m).  A value
            of ``0.0`` (default) disables repulsion.
    """

    def __init__(
        self,
        vehicle: DubinsVehicle,
        controller: PurePursuitController,
        cruise_speed: float = 1.0,
        curvature_gain: float = 0.0,
        occupancy: Optional["Occupancy"] = None,
        repulsion_gain: float = 0.0,
    ) -> None:
        """Initialize TrackingLoop.

        Args:
            vehicle: Kinematic vehicle model.
            controller: Pure pursuit path-tracking controller.
            cruise_speed: Desired forward speed (m/s).
            curvature_gain: Speed-modulation gain (m).  Set to ``0.0`` to
                keep a constant cruise speed.  Positive values slow the
                vehicle on curves: ``v = cruise_speed / (1 + gain * |κ|)``.
            occupancy: Optional occupancy map used to compute obstacle
                repulsion.  When ``None`` (default) or when *repulsion_gain*
                is ``0.0``, no repulsion correction is applied.
            repulsion_gain: Obstacle-repulsion turn-rate gain (rad/m).
                Positive values add a corrective turn when the vehicle
                approaches obstacles.  Typical range: ``0.5``–``3.0``.
        """
        self.vehicle = vehicle
        self.controller = controller
        self.cruise_speed = cruise_speed
        self.curvature_gain = curvature_gain
        self._occupancy = occupancy
        self.repulsion_gain = repulsion_gain
        self._history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def metrics(self) -> dict[str, Any] | None:
        """Most recent step metrics, or ``None`` if no steps have been run."""
        return self._history[-1] if self._history else None

    @property
    def history(self) -> list[dict[str, Any]]:
        """Full per-step metrics history (read-only copy)."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _repulsion_turn_rate(self, x: float, y: float, theta: float) -> float:
        """Compute an APF obstacle-repulsion turn-rate correction.

        Returns an additive turn-rate (rad/s) that steers the vehicle away
        from the nearest obstacle when it is within ``2 × clearance`` meters.
        The magnitude follows the standard APF formula::

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
            Turn-rate correction in rad/s; ``0.0`` when outside influence
            radius or when no occupancy map is configured.
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

        # Steer away: if obstacle is to the left (lateral > 0) → turn right (Δω < 0)
        lateral_sign = math.copysign(1.0, lateral)
        return -magnitude * lateral_sign

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def step(
        self, path: list[tuple[float, float]], dt: float = 0.1
    ) -> dict[str, Any]:
        """Run one tracking iteration.

        Queries the controller for speed and turn-rate commands given the
        current vehicle pose and reference path, applies them to the vehicle
        model, then records and returns the resulting metrics.

        When an occupancy map and a positive repulsion gain are configured,
        an obstacle-avoidance correction is blended into the turn-rate
        command before the vehicle is integrated.

        Args:
            path: Reference path as an ordered list of ``(x, y)`` waypoints.
            dt: Integration time step (s).

        Returns:
            Dictionary with keys:

            - ``cross_track_error``: signed perpendicular distance from
              vehicle to nearest path segment (meters).
            - ``heading_error``: vehicle heading minus path tangent, wrapped
              to ``(−π, π]`` (radians).
            - ``pose``: current vehicle pose ``(x, y, heading)``.
            - ``speed``: current vehicle speed (m/s).
            - ``turn_rate``: current vehicle turn rate (rad/s).
            - ``curvature``: pure pursuit curvature used this step (rad/m).
            - ``repulsion_turn_rate``: obstacle-repulsion correction added
              to the turn-rate command (rad/s).  Zero when repulsion is
              disabled.
        """
        pose = self.vehicle.pose
        speed_ref = self.cruise_speed / (
            1.0 + self.curvature_gain * abs(self.controller.curvature)
        )
        speed_cmd, turn_rate_cmd = self.controller.track(pose, path, speed_ref)

        # Blend in obstacle repulsion correction.
        x, y, theta = pose
        repulsion = self._repulsion_turn_rate(x, y, theta)
        turn_rate_cmd += repulsion

        self.vehicle.step(speed_cmd, turn_rate_cmd, dt)
        entry: dict[str, Any] = {
            "cross_track_error": self.controller.cross_track_error,
            "heading_error": self.controller.heading_error,
            "pose": self.vehicle.pose,
            "speed": self.vehicle.speed,
            "turn_rate": self.vehicle.turn_rate,
            "curvature": self.controller.curvature,
            "repulsion_turn_rate": repulsion,
        }
        self._history.append(entry)
        return entry

    def run(
        self, path: list[tuple[float, float]], steps: int, dt: float = 0.1
    ) -> list[dict[str, Any]]:
        """Run multiple tracking steps.

        Args:
            path: Reference path as an ordered list of ``(x, y)`` waypoints.
            steps: Number of steps to simulate.
            dt: Integration time step (s).

        Returns:
            List of per-step metric dictionaries (same schema as :meth:`step`).
        """
        return [self.step(path, dt) for _ in range(steps)]
