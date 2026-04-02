"""Graph: base graph data structure."""

# Copyright 2026 Alexandre Loeblein Heinen

from __future__ import annotations


class Graph:
    """Representation of a graph G = (V, E).

    Mathematically, a graph is defined as G = (V, E) where V is a set of
    vertices and E is a set of edges.
    """

    class Node:
        """A node in the graph."""

        def __init__(self) -> None:
            """Initialize an empty node."""
            pass

    class Edge:
        """An undirected edge in the graph."""

        def __init__(self, node_0: "Graph.Node", node_1: "Graph.Node") -> None:
            """Initialize an edge between two nodes.

            Args:
                node_0: The first node.
                node_1: The second node.
            """
            pass
