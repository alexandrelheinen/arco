"""DStarLite: public API wrapper for the D* path planner (stub)."""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np


class DStarLite:
    """Public API wrapper for D* planner (stub — not yet implemented)."""

    def __init__(self, grid: np.ndarray) -> None:
        """Initialize the DStarLite wrapper.

        Args:
            grid: A numpy array where 0=free and 1=occupied.
        """

    def search(self, start: Any, goal: Any) -> Optional[List[Any]]:
        """Search for a path from start to goal (not yet implemented).

        Args:
            start: The start node index.
            goal: The goal node index.

        Returns:
            A list of node indices from start to goal, or None if no path exists.

        Raises:
            NotImplementedError: D* planner is not yet implemented.
        """
        raise NotImplementedError("D* planner not yet implemented.")
