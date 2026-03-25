"""Graph data structure for mapping problems."""

# Copyright 2026 Alexandre Loeblein Heinen


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
