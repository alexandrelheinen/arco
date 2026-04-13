"""TrackingLoop: local control loop for route following with bounded dynamics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from arco.guidance.vehicle import DubinsVehicle

from .pure_pursuit import PurePursuitController


class TrackingLoop:
    """Local tracking loop combining a vehicle model and a path controller.

    Closes the feedback loop between a :class:`~arco.guidance.vehicle.DubinsVehicle`
    kinematic model and a :class:`PurePursuitController`.  Each call to
    :meth:`step` issues pure pursuit commands to the vehicle and records
    cross-track error, heading error, pose, speed, and turn rate for later
    analysis.

    Attributes:
        vehicle: Kinematic vehicle model.
        controller: Pure pursuit path-tracking controller.
        cruise_speed: Desired forward speed passed to the controller (m/s).
        curvature_gain: Curvature-to-speed scaling factor (m).  Speed is
            modulated as ``v = cruise_speed / (1 + curvature_gain * |κ|)``
            where *κ* is the pure pursuit curvature from the previous step.
            A value of ``0.0`` (default) disables modulation.
    """

    def __init__(
        self,
        vehicle: DubinsVehicle,
        controller: PurePursuitController,
        cruise_speed: float = 1.0,
        curvature_gain: float = 0.0,
    ) -> None:
        """Initialize TrackingLoop.

        Args:
            vehicle: Kinematic vehicle model.
            controller: Pure pursuit path-tracking controller.
            cruise_speed: Desired forward speed (m/s).
            curvature_gain: Speed-modulation gain (m).  Set to ``0.0`` to
                keep a constant cruise speed.  Positive values slow the
                vehicle on curves: ``v = cruise_speed / (1 + gain * |κ|)``.
        """
        self.vehicle = vehicle
        self.controller = controller
        self.cruise_speed = cruise_speed
        self.curvature_gain = curvature_gain
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
    # Simulation
    # ------------------------------------------------------------------

    def step(
        self, path: list[tuple[float, float]], dt: float = 0.1
    ) -> dict[str, Any]:
        """Run one tracking iteration.

        Queries the controller for speed and turn-rate commands given the
        current vehicle pose and reference path, applies them to the vehicle
        model, then records and returns the resulting metrics.

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
        """
        pose = self.vehicle.pose
        speed_ref = self.cruise_speed / (
            1.0 + self.curvature_gain * abs(self.controller.curvature)
        )
        speed_cmd, turn_rate_cmd = self.controller.track(pose, path, speed_ref)
        self.vehicle.step(speed_cmd, turn_rate_cmd, dt)
        entry: dict[str, Any] = {
            "cross_track_error": self.controller.cross_track_error,
            "heading_error": self.controller.heading_error,
            "pose": self.vehicle.pose,
            "speed": self.vehicle.speed,
            "turn_rate": self.vehicle.turn_rate,
            "curvature": self.controller.curvature,
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
