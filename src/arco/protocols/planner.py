"""PlannerLike: duck-typed contract for pipeline planners."""

from __future__ import annotations

from typing import List, Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class PlannerLike(Protocol):
    """Continuous planner stage: ``plan(start, goal) → path | None``."""

    def plan(
        self, start: np.ndarray, goal: np.ndarray
    ) -> Optional[List[np.ndarray]]:
        """Plan a path from *start* to *goal*."""
