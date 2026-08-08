"""TelemetryPublisher: duck-typed contract for planner telemetry sinks."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TelemetryPublisher(Protocol):
    """Sink for planner telemetry snapshots.

    The default implementation writes JSON to a temp file.  Inject a
    no-op or custom publisher to disable or redirect telemetry.

    Args accepted by ``__call__`` match
    :class:`~arco.planning.continuous.telemetry.PlannerTelemetry`.
    """

    def __call__(self, telemetry: Any) -> None:
        """Publish one telemetry snapshot."""
