"""Graph data structure for mapping problems."""

# Copyright 2026 Alexandre Loeblein Heinen

from __future__ import annotations

import math
from typing import Dict, Iterator, List, Optional, Tuple


class Graph:
    """
    Representation of a graph.

    Mathematically, a graph is defined as
        G = (V, E)
    where V is a set of vertices and E is a set of edges.
    """

    class Node:
        """A node in the graph."""

        def __init__(self):
            """Initialize an empty node."""
            pass

    class Edge:
        """An undirected edge in the graph."""

        def __init__(self, node_0, node_1):
            """
            Initialize an edge between two nodes.

            Args:
                node_0: The first node.
                node_1: The second node.
            """
            pass


class OrientedGraph(Graph):
    """
    Mathematical representation of an oriented graph
        G = (V, E)
    where V is a set of vertices and E is a set of directed edges (arcs).
    """

    class Arc(Graph.Edge):
        """
        A directed edge (arc) in the graph.
        """

        def __init__(self, node_from, node_to):
            """
            Initialize an arc between two nodes.

            Args:
                node_from (Graph.Node): The starting node.
                node_to (Graph.Node): The ending node.
            """
            super().__init__(node_from, node_to)


class WeightedGraph(Graph):
    """
    Weighted undirected graph with 2D node positions.

    Nodes are identified by integer IDs and carry (x, y) positions.
    Edges are weighted; if no weight is supplied when adding an edge the
    Euclidean distance between the two node positions is used.

    Implements the same ``neighbors`` / ``distance`` interface as
    :class:`~arco.mapping.grid.base.Grid` so that
    :class:`~arco.planning.discrete.astar.AStarPlanner` can operate on it
    without modification.
    """

    def __init__(self) -> None:
        """Initialize an empty weighted graph."""
        super().__init__()
        self._positions: Dict[int, Tuple[float, float]] = {}
        self._adjacency: Dict[int, List[Tuple[int, float]]] = {}

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def add_node(self, node_id: int, x: float, y: float) -> None:
        """Add a node with a 2D position.

        Args:
            node_id: Unique integer identifier for the node.
            x: X coordinate of the node.
            y: Y coordinate of the node.
        """
        self._positions[node_id] = (x, y)
        if node_id not in self._adjacency:
            self._adjacency[node_id] = []

    def add_edge(
        self, node_a: int, node_b: int, weight: Optional[float] = None
    ) -> None:
        """Add an undirected weighted edge between two nodes.

        Args:
            node_a: ID of the first node.
            node_b: ID of the second node.
            weight: Edge weight. Defaults to the Euclidean distance between
                the two node positions when *None*.
        """
        if weight is None:
            weight = self._euclidean(node_a, node_b)
        self._adjacency[node_a].append((node_b, weight))
        self._adjacency[node_b].append((node_a, weight))

    # ------------------------------------------------------------------
    # Planner interface
    # ------------------------------------------------------------------

    def neighbors(self, node_id: int) -> Iterator[int]:
        """Yield the IDs of all nodes directly connected to *node_id*.

        Args:
            node_id: ID of the query node.

        Yields:
            Neighbour node IDs.
        """
        for neighbour, _ in self._adjacency.get(node_id, []):
            yield neighbour

    def distance(self, node_a: int, node_b: int) -> float:
        """Return the edge weight between two adjacent nodes, falling back to
        the Euclidean distance when no direct edge exists.

        Args:
            node_a: ID of the first node.
            node_b: ID of the second node.

        Returns:
            Edge weight as a float.
        """
        for neighbour, weight in self._adjacency.get(node_a, []):
            if neighbour == node_b:
                return weight
        return self._euclidean(node_a, node_b)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def position(self, node_id: int) -> Tuple[float, float]:
        """Return the (x, y) position of a node.

        Args:
            node_id: ID of the node.

        Returns:
            (x, y) tuple.
        """
        return self._positions[node_id]

    @property
    def nodes(self) -> List[int]:
        """Return a list of all node IDs."""
        return list(self._positions.keys())

    @property
    def edges(self) -> List[Tuple[int, int, float]]:
        """Return all edges as ``(node_a, node_b, weight)`` triples.

        Each undirected edge is returned only once (a < b convention).
        """
        seen: set = set()
        result: List[Tuple[int, int, float]] = []
        for a, neighbours in self._adjacency.items():
            for b, w in neighbours:
                key = (min(a, b), max(a, b))
                if key not in seen:
                    seen.add(key)
                    result.append((key[0], key[1], w))
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _euclidean(self, node_a: int, node_b: int) -> float:
        xa, ya = self._positions[node_a]
        xb, yb = self._positions[node_b]
        return math.hypot(xa - xb, ya - yb)
