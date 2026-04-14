"""Planner telemetry: stop-criterion data structures and temp-file I/O.

# TODO(IPC-MIDDLEWARE): This file-based IPC is a stopgap.  The full
# architecture requires a pub/sub middleware where every pipeline module
# publishes its internal state and any observer (loading screen, dashboard,
# external monitor) subscribes.  See docs/ROADMAP.md § "IPC & Telemetry
# Middleware" for the open design task.

Each planning algorithm writes a :class:`PlannerTelemetry` snapshot to a
JSON temp-file at :data:`DEFAULT_TELEMETRY_PATH` every
:data:`TELEMETRY_WRITE_INTERVAL` iterations so that the loading screen can
poll it and show live stop-criteria progress.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

TELEMETRY_WRITE_INTERVAL: int = 100
DEFAULT_TELEMETRY_PATH: Path = (
    Path(tempfile.gettempdir()) / "arco_planner_telemetry.json"
)


@dataclass
class StopCriterion:
    """A single named stop criterion with its current and threshold values.

    Args:
        name: Human-readable criterion label (e.g. ``"iterations"``).
        current: Current measured value.
        threshold: Target threshold value.
        condition: Comparison operator string (``"<"``, ``"≤"``, ``"≥"``,
            ``">"``, ``"="``).
    """

    name: str
    current: float
    threshold: float
    condition: str

    def satisfied(self) -> bool:
        """Return True if the criterion is currently met.

        Returns:
            True when the condition expression ``current <op> threshold``
            evaluates to True.
        """
        ops = {
            "<": lambda a, b: a < b,
            "≤": lambda a, b: a <= b,
            "<=": lambda a, b: a <= b,
            ">": lambda a, b: a > b,
            "≥": lambda a, b: a >= b,
            ">=": lambda a, b: a >= b,
            "=": lambda a, b: a == b,
            "==": lambda a, b: a == b,
        }
        fn = ops.get(self.condition)
        if fn is None:
            return False
        return bool(fn(self.current, self.threshold))


@dataclass
class PlannerTelemetry:
    """A telemetry snapshot written by a planner at regular intervals.

    Args:
        algorithm: Name of the planning algorithm (e.g. ``"RRT*"``).
        step_name: Human-readable description of the current planning phase.
        iteration: Current iteration index.
        max_iterations: Maximum allowed iterations.
        best_dist_to_goal: Smallest distance to goal observed so far.
        criteria: List of stop criteria with current values.
    """

    algorithm: str
    step_name: str
    iteration: int
    max_iterations: int
    best_dist_to_goal: float
    criteria: list[StopCriterion] = field(default_factory=list)


def write_telemetry(
    telemetry: PlannerTelemetry,
    path: Path = DEFAULT_TELEMETRY_PATH,
) -> None:
    """Write *telemetry* as JSON to *path* atomically.

    The write is performed via a sibling ``.tmp`` file that is then renamed
    over the target to avoid partial reads.  Any I/O error is silently
    swallowed so this function never raises.

    Args:
        telemetry: The snapshot to persist.
        path: Destination file path (defaults to
            :data:`DEFAULT_TELEMETRY_PATH`).
    """
    try:
        data = asdict(telemetry)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(data), encoding="utf-8")
        tmp_path.replace(path)
    except Exception:  # noqa: BLE001
        pass


def read_telemetry(
    path: Path = DEFAULT_TELEMETRY_PATH,
) -> Optional[PlannerTelemetry]:
    """Read and parse the telemetry snapshot at *path*.

    Args:
        path: Source file path (defaults to :data:`DEFAULT_TELEMETRY_PATH`).

    Returns:
        A :class:`PlannerTelemetry` instance, or ``None`` on any error.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        criteria = [StopCriterion(**c) for c in data.get("criteria", [])]
        return PlannerTelemetry(
            algorithm=data["algorithm"],
            step_name=data["step_name"],
            iteration=data["iteration"],
            max_iterations=data["max_iterations"],
            best_dist_to_goal=data["best_dist_to_goal"],
            criteria=criteria,
        )
    except Exception:  # noqa: BLE001
        return None
