"""Steerer: duck-typed contract for continuous planner extension."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Steerer(Protocol):
    """Steering law used by RRT*/SST (and optional pruner feasibility).

    ``__call__(from_pt, to_pt) → new_state`` matches the default
    step-size-limited straight-line steerer.
    """

    def __call__(self, from_pt: np.ndarray, to_pt: np.ndarray) -> np.ndarray:
        """Steer from *from_pt* toward *to_pt* by at most one step."""
