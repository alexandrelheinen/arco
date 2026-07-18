"""MPCTrackingLoop: drop-in parallel to TrackingLoop using an MPC tracker."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from arco.guidance.vehicle import DubinsVehicle

from arco.control.mpc.base import MPCTracker


class MPCTrackingLoop:
    """Local tracking loop driven by a :class:`MPCTracker`.

    Mirrors :class:`~arco.control.tracking.TrackingLoop` metrics for
    drop-in instrumentation while keeping obstacle avoidance inside the
    optimizer (no APF repulsion blend).

    Attributes:
        vehicle: Kinematic vehicle model.
        tracker: Receding-horizon MPC tracker.
        cruise_speed: Nominal cruise speed used for progress weighting
            inside the tracker config (informational mirror).
    """

    def __init__(
        self,
        vehicle: DubinsVehicle,
        tracker: MPCTracker,
        cruise_speed: float = 1.0,
    ) -> None:
        """Initialize MPCTrackingLoop.

        Args:
            vehicle: Kinematic vehicle model.
            tracker: Configured MPC tracker (reference set later via
                :meth:`step` or :meth:`MPCTracker.set_reference`).
            cruise_speed: Desired forward speed (m/s), stored for
                metrics compatibility with :class:`TrackingLoop`.
        """
        self.vehicle = vehicle
        self.tracker = tracker
        self.cruise_speed = cruise_speed
        self._history: list[dict[str, Any]] = []
        self._reference_signature: tuple[tuple[float, float], ...] | None = (
            None
        )

    @property
    def metrics(self) -> dict[str, Any] | None:
        """Most recent step metrics, or ``None`` if no steps have been run."""
        return self._history[-1] if self._history else None

    @property
    def history(self) -> list[dict[str, Any]]:
        """Full per-step metrics history (read-only copy)."""
        return list(self._history)

    def _ensure_reference(self, path: list[tuple[float, float]]) -> None:
        signature = tuple((float(x), float(y)) for x, y in path)
        if signature != self._reference_signature:
            self.tracker.set_reference(path)
            self._reference_signature = signature

    def step(
        self, path: list[tuple[float, float]], dt: float = 0.1
    ) -> dict[str, Any]:
        """Run one MPC tracking iteration.

        Args:
            path: Reference path as an ordered list of ``(x, y)`` waypoints.
            dt: Integration time step (s).

        Returns:
            Dictionary with keys compatible with
            :meth:`~arco.control.tracking.TrackingLoop.step`, plus
            ``mpc_*`` diagnostics from :class:`MPCStepResult`.
        """
        self._ensure_reference(path)
        pose = self.vehicle.pose
        result = self.tracker.step(
            pose,
            speed=self.vehicle.speed,
            turn_rate=self.vehicle.turn_rate,
            dt=dt,
        )
        self.vehicle.step(result.speed_cmd, result.turn_rate_cmd, dt)
        entry: dict[str, Any] = {
            "cross_track_error": result.cross_track_error,
            "heading_error": result.heading_error,
            "pose": self.vehicle.pose,
            "speed": self.vehicle.speed,
            "turn_rate": self.vehicle.turn_rate,
            # TrackingLoop compatibility: no pure-pursuit curvature / APF.
            "curvature": 0.0,
            "repulsion_turn_rate": 0.0,
            "mpc_progress": result.progress,
            "mpc_predicted_clearance_min": result.predicted_clearance_min,
            "mpc_solver_success": result.solver_success,
            "mpc_solver_status": result.solver_status,
            "mpc_solve_time_s": result.solve_time_s,
            "mpc_cost": result.cost,
            "mpc_speed_cmd": result.speed_cmd,
            "mpc_turn_rate_cmd": result.turn_rate_cmd,
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
