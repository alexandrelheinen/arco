"""Base class for discrete planners operating on graphs (including grids)."""

from __future__ import annotations

from typing import Any

from arco.planning.cost import PlannerCost


class DiscretePlanner(PlannerCost):
    """
    Base class for discrete planners operating on graphs (including grids).

    Accepts any Graph (e.g., Grid, Occupancy, custom Graph).
    Inherits default :meth:`distance` / :meth:`heuristic` from
    :class:`~arco.planning.cost.PlannerCost`, and overrides them to
    delegate to the attached graph.
    """

    graph: Any

    def __init__(self, graph: Any) -> None:
        """
        Initialize the planner with a graph.

        Args:
            graph: The graph structure (Grid, Occupancy, or custom Graph).
                Must expose ``distance``.  Optionally exposes ``heuristic``.
        """
        self.graph = graph

    def distance(self, state_a: Any, state_b: Any) -> float:
        """Return the graph edge cost between two nodes.

        Args:
            state_a: Origin node.
            state_b: Destination node.

        Returns:
            Edge cost from ``graph.distance``.
        """
        return float(self.graph.distance(state_a, state_b))

    def heuristic(self, state_a: Any, state_b: Any) -> float:
        """Return a remaining-cost estimate between two nodes.

        Uses ``graph.heuristic`` when available; otherwise falls back to
        :meth:`distance`.

        Args:
            state_a: Current node.
            state_b: Goal node.

        Returns:
            Heuristic cost estimate.
        """
        if hasattr(self.graph, "heuristic"):
            return float(self.graph.heuristic(state_a, state_b))
        return self.distance(state_a, state_b)
