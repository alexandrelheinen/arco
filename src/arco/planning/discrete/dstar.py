"""DStarPlanner: D* discrete path planner (stub)."""

from __future__ import annotations

from typing import Any, List, Optional

from .base import DiscretePlanner


class DStarPlanner(DiscretePlanner):
    """D* path planner for dynamic replanning (stub).

    D* (Dynamic A*) supports incremental replanning when the environment
    changes. This implementation is a placeholder.
    """

    def plan(self, start: Any, goal: Any) -> Optional[List[Any]]:
        """Plan a path from start to goal (not yet implemented).

        Args:
            start: The start node.
            goal: The goal node.

        Returns:
            A list of nodes from start to goal, or None if no path exists.

        Raises:
            NotImplementedError: D* planner is not yet implemented.
        """
        raise NotImplementedError("D* planner not yet implemented.")
