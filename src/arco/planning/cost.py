"""Shared default cost primitives for path planners."""

from __future__ import annotations

from typing import Any

import numpy as np


class PlannerCost:
    """Default distance and heuristic cost functions for planners.

    A*, RRT*, and SST inherit these methods through their discrete or
    continuous base classes.  Override :meth:`distance` and/or
    :meth:`heuristic` in a subclass to customize edge cost and the
    remaining-cost estimate without rewriting the search algorithm.

    Default metric: Euclidean distance between array-like states.
    The default heuristic equals :meth:`distance` and is therefore
    admissible for Euclidean path costs.
    """

    def distance(self, state_a: Any, state_b: Any) -> float:
        """Return the transition cost between two states.

        Args:
            state_a: Origin state (array-like).
            state_b: Destination state (array-like).

        Returns:
            Non-negative Euclidean distance between the states.
        """
        a = np.asarray(state_a, dtype=float).reshape(-1)
        b = np.asarray(state_b, dtype=float).reshape(-1)
        return float(np.linalg.norm(b - a))

    def heuristic(self, state_a: Any, state_b: Any) -> float:
        """Return an admissible estimate of remaining cost from a to b.

        Args:
            state_a: Current state (array-like).
            state_b: Goal state (array-like).

        Returns:
            Estimated remaining cost.  Defaults to :meth:`distance`.
        """
        return self.distance(state_a, state_b)
