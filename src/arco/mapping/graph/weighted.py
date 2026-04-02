"""WeightedGraph: generic weighted undirected graph data structure."""

# Copyright 2026 Alexandre Loeblein Heinen

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple

from .base import Graph


class WeightedGraph(Graph):
    """Generic weighted undirected graph.

    Nodes are identified by integer IDs.  Edges carry explicit numeric
    weights.  No spatial or positional concept is embedded in this class;
    positional data belongs in :class:`~arco.mapping.graph.CartesianGraph`.

    Implements the ``neighbors`` / ``distance`` interface expected by
    :class:`~arco.planning.discrete.astar.AStarPlanner`.
    """

    def __init__(self) -> None:
        """Initialize an empty weighted graph."""
        super().__init__()
        self._adjacency: Dict[int, List[Tuple[int, float]]] = {}

    def add_node(self, node_id: int) -> None:
        """Register a node in the graph.

        Args:
            node_id: Unique integer identifier for the node.
        """
        if node_id not in self._adjacency:
            self._adjacency[node_id] = []

    def add_edge(self, node_a: int, node_b: int, weight: float) -> None:
        """Add an undirected weighted edge between two nodes.

        Args:
            node_a: ID of the first node.
            node_b: ID of the second node.
            weight: Edge weight.
        """
        self._adjacency[node_a].append((node_b, weight))
        self._adjacency[node_b].append((node_a, weight))

    def neighbors(self, node_id: int) -> Iterator[int]:
        """Yield the IDs of all nodes directly connected to *node_id*.

        Args:
            node_id: ID of the query node.

        Yields:
            Neighbor node IDs.
        """
        for neighbor, _ in self._adjacency.get(node_id, []):
            yield neighbor

    def distance(self, node_a: int, node_b: int) -> float:
        """Return the edge weight between two adjacent nodes.

        Args:
            node_a: ID of the first node.
            node_b: ID of the second node.

        Returns:
            Edge weight as a float.

        Raises:
            KeyError: If no edge exists between *node_a* and *node_b*.
        """
        for neighbor, weight in self._adjacency.get(node_a, []):
            if neighbor == node_b:
                return weight
        raise KeyError(f"No edge between nodes {node_a!r} and {node_b!r}.")

    @property
    def nodes(self) -> List[int]:
        """Return a list of all node IDs."""
        return list(self._adjacency.keys())

    @property
    def edges(self) -> List[Tuple[int, int, float]]:
        """Return all edges as ``(node_a, node_b, weight)`` triples.

        Each undirected edge is returned only once (a < b convention).
        """
        seen: set = set()
        result: List[Tuple[int, int, float]] = []
        for a, neighbors in self._adjacency.items():
            for b, w in neighbors:
                key = (min(a, b), max(a, b))
                if key not in seen:
                    seen.add(key)
                    result.append((key[0], key[1], w))
        return result
