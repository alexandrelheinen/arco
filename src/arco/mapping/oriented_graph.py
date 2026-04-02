"""OrientedGraph: directed graph data structure."""

from __future__ import annotations

from .graph import Graph


class OrientedGraph(Graph):
    """Mathematical representation of an oriented graph G = (V, E).

    V is a set of vertices and E is a set of directed edges (arcs).
    """

    class Arc(Graph.Edge):
        """A directed edge (arc) in the oriented graph."""

        def __init__(self, node_from: Graph.Node, node_to: Graph.Node) -> None:
            """Initialize an arc between two nodes.

            Args:
                node_from: The starting node.
                node_to: The ending node.
            """
            super().__init__(node_from, node_to)
