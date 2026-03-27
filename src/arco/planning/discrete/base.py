
"""Base class for discrete planners operating on graphs (including grids)."""

from __future__ import annotations
from typing import Any


class DiscretePlanner:
    """
    Base class for discrete planners operating on graphs (including grids).

    Accepts any Graph (e.g., Grid, Occupancy, custom Graph).
    """

    graph: Any

    def __init__(self, graph: Any) -> None:
        """
        Initialize the planner with a graph.

        Args:
            graph: The graph structure (Grid, Occupancy, or custom Graph).
        """
        self.graph = graph
