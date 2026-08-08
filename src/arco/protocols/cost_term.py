"""CostTerm: duck-typed contract for optimizer cost contributions."""

from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class CostTerm(Protocol):
    """One term of a composite trajectory cost.

    ``__call__(context) → float`` receives a dict with unpacked
    durations, waypoints, and optimizer settings (see
    :class:`~arco.planning.continuous.optimizer.TrajectoryOptimizer`).
    """

    name: str

    def __call__(self, context: Dict[str, Any]) -> float:
        """Evaluate this term given the optimization context."""
