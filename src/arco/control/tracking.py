"""TrackingLoop: local control loop for route following with bounded dynamics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from .avoidance import ArtificialPotentialField

if TYPE_CHECKING:
    from arco.mapping.occupancy import Occupancy
    from arco.protocols.avoidance import AvoidanceStrategy
    from arco.protocols.path_tracker import PathTracker
    from arco.protocols.vehicle import VehicleModel


class TrackingLoop:
    """Local tracking loop combining a vehicle model and a path controller.

    Closes the feedback loop between a
    :class:`~arco.protocols.vehicle.VehicleModel`-compatible kinematic
    model and a :class:`~arco.protocols.path_tracker.PathTracker`-compatible
    controller.  Each call to :meth:`step` issues tracking commands to the
    vehicle and records cross-track error, heading error, pose, speed, and
    turn rate for later analysis.

    When an *occupancy* map and a positive *repulsion_gain* are provided
    (and *avoidance* is omitted), a default
    :class:`~arco.control.avoidance.ArtificialPotentialField` correction is
    blended into the turn-rate command at every step.  Pass an explicit
    *avoidance* strategy to replace the inline APF; pass ``None`` with
    non-positive *repulsion_gain* (or no occupancy) to disable repulsion.

    Attributes:
        vehicle: Kinematic vehicle model.
        controller: Path-tracking controller.
        cruise_speed: Desired forward speed passed to the controller (m/s).
        curvature_gain: Curvature-to-speed scaling factor (m).  Speed is
            modulated as ``v = cruise_speed / (1 + curvature_gain * |κ|)``
            where *κ* is the tracker curvature from the previous step.
            A value of ``0.0`` (default) disables modulation.
        repulsion_gain: Obstacle-repulsion turn-rate gain (rad/m) used when
            building the default APF.  A value of ``0.0`` (default)
            disables repulsion unless a custom *avoidance* is supplied.
    """

    def __init__(
        self,
        vehicle: "VehicleModel",
        controller: "PathTracker",
        cruise_speed: float = 1.0,
        curvature_gain: float = 0.0,
        occupancy: Optional["Occupancy"] = None,
        repulsion_gain: float = 0.0,
        avoidance: Optional["AvoidanceStrategy"] = None,
    ) -> None:
        """Initialize TrackingLoop.

        Args:
            vehicle: Kinematic vehicle model satisfying
                :class:`~arco.protocols.vehicle.VehicleModel`.
            controller: Path tracker satisfying
                :class:`~arco.protocols.path_tracker.PathTracker`.
            cruise_speed: Desired forward speed (m/s).
            curvature_gain: Speed-modulation gain (m).  Set to ``0.0`` to
                keep a constant cruise speed.  Positive values slow the
                vehicle on curves: ``v = cruise_speed / (1 + gain * |κ|)``.
            occupancy: Optional occupancy map used to build the default
                APF when *avoidance* is ``None``.  Ignored when a custom
                *avoidance* is provided.  When ``None`` (default) or when
                *repulsion_gain* is ``0.0``, no default repulsion is
                applied.
            repulsion_gain: Obstacle-repulsion turn-rate gain (rad/m) for
                the default APF.  Positive values add a corrective turn
                when the vehicle approaches obstacles.  Typical range:
                ``0.5``–``3.0``.  Kept for backwards compatibility.
            avoidance: Optional
                :class:`~arco.protocols.avoidance.AvoidanceStrategy`.
                When provided, it is used instead of the default APF.
                When ``None`` and *repulsion_gain* > 0 with *occupancy*
                set, an :class:`ArtificialPotentialField` is constructed
                with the same behaviour as the historical inline APF.
        """
        self.vehicle = vehicle
        self.controller = controller
        self.cruise_speed = cruise_speed
        self.curvature_gain = curvature_gain
        self._occupancy = occupancy
        self.repulsion_gain = repulsion_gain
        if avoidance is not None:
            self._avoidance: Optional["AvoidanceStrategy"] = avoidance
        elif occupancy is not None and repulsion_gain > 0.0:
            self._avoidance = ArtificialPotentialField(
                occupancy=occupancy,
                repulsion_gain=repulsion_gain,
            )
        else:
            self._avoidance = None
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
        """Compute an obstacle-repulsion turn-rate correction.

        Delegates to the configured :class:`~arco.protocols.avoidance.AvoidanceStrategy`
        (default APF or an injected strategy).  Returns ``0.0`` when
        avoidance is disabled.

        Args:
            x: Vehicle x-position in world frame (m).
            y: Vehicle y-position in world frame (m).
            theta: Vehicle heading in radians.

        Returns:
            Turn-rate correction in rad/s; ``0.0`` when avoidance is
            disabled.
        """
        if self._avoidance is None:
            return 0.0
        return float(self._avoidance(x, y, theta))

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

        When an avoidance strategy is configured (explicitly or via the
        default APF from occupancy / repulsion_gain), an obstacle-avoidance
        correction is blended into the turn-rate command before the vehicle
        is integrated.

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
            - ``curvature``: tracker curvature used this step (rad/m).
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
