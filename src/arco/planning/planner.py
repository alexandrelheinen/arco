"""Planner: base class for all planners."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, List, Optional

from arco.mapping.graph import Graph


class Planner:
    """Base class for all path planners."""

    def __init__(self, graph: Graph) -> None:
        """Initialize the planner with a graph.

        Args:
            graph: The graph structure to plan on.
        """
        self._graph = graph

    @abstractmethod
    def plan(self, start: Any, goal: Any) -> Optional[List[Any]]:
        """Plan a path from start to goal.

        Args:
            start: The start node.
            goal: The goal node.

        Returns:
            A list of nodes from start to goal, or None if no path exists.
        """
