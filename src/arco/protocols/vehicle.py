"""VehicleModel: duck-typed contract for SE(2) kinematic vehicles."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class VehicleModel(Protocol):
    """SE(2) vehicle surface used by tracking loops.

    Matches :class:`~arco.guidance.vehicle.DubinsVehicle` pose/step API.
    """

    @property
    def pose(self) -> tuple[float, float, float]:
        """Current pose as ``(x, y, heading)``."""

    @property
    def speed(self) -> float:
        """Current forward speed (m/s)."""

    @property
    def turn_rate(self) -> float:
        """Current turn rate (rad/s)."""

    def step(self, speed_cmd: float, turn_rate_cmd: float, dt: float) -> None:
        """Integrate one control step of duration *dt*."""
