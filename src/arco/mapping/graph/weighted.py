"""WeightedGraph: weighted undirected graph data structure."""

from __future__ import annotations

import math
from typing import Dict, Iterator, List, Optional, Tuple

from .base import Graph


class WeightedGraph(Graph):
    """Weighted undirected graph with 2D node positions.

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

        Falls back to the Euclidean distance when no direct edge exists.

        Args:
            node_a: ID of the first node.
            node_b: ID of the second node.

        Returns:
            Edge weight as a float.
        """
        for neighbor, weight in self._adjacency.get(node_a, []):
            if neighbor == node_b:
                return weight
        return self._euclidean(node_a, node_b)

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
        for a, neighbors in self._adjacency.items():
            for b, w in neighbors:
                key = (min(a, b), max(a, b))
                if key not in seen:
                    seen.add(key)
                    result.append((key[0], key[1], w))
        return result

    def _euclidean(self, node_a: int, node_b: int) -> float:
        xa, ya = self._positions[node_a]
        xb, yb = self._positions[node_b]
        return math.hypot(xa - xb, ya - yb)

    def heuristic(self, node_a: int, node_b: int) -> float:
        """Return the Euclidean distance between two nodes.

        This method provides a heuristic for A* search that is admissible
        (never overestimates) and consistent for path planning on road graphs.

        Args:
            node_a: ID of the first node.
            node_b: ID of the second node.

        Returns:
            Euclidean distance as a float.
        """
        return self._euclidean(node_a, node_b)

    def find_nearest_node(
        self, x: float, y: float, max_radius: Optional[float] = None
    ) -> Optional[int]:
        """Find the node closest to the given position.

        Args:
            x: X coordinate of the query position.
            y: Y coordinate of the query position.
            max_radius: Maximum search radius. If specified, only returns
                nodes within this distance. Returns None if no node is found
                within the radius.

        Returns:
            ID of the nearest node, or None if no node exists within max_radius.
        """
        if not self._positions:
            return None

        nearest_node = None
        nearest_dist = float("inf")

        for node_id, (nx, ny) in self._positions.items():
            dist = math.hypot(x - nx, y - ny)
            if dist < nearest_dist:
                if max_radius is None or dist <= max_radius:
                    nearest_dist = dist
                    nearest_node = node_id

        return nearest_node

    def project_to_nearest_edge(
        self, x: float, y: float, max_radius: Optional[float] = None
    ) -> Optional[Tuple[Tuple[float, float], int, int, float]]:
        """Project a point onto the nearest edge of the graph.

        Finds the closest point on any edge to the query position by computing
        the perpendicular projection onto each edge line segment.

        Args:
            x: X coordinate of the query position.
            y: Y coordinate of the query position.
            max_radius: Maximum search radius. If specified, only considers
                edges where the projected point is within this distance.

        Returns:
            A tuple containing:
                - (proj_x, proj_y): The projected point coordinates
                - node_a: ID of the first endpoint of the nearest edge
                - node_b: ID of the second endpoint of the nearest edge
                - distance: Distance from the query point to the projection
            Returns None if no edge is found within max_radius.
        """
        if not self.edges:
            return None

        nearest_projection = None
        nearest_dist = float("inf")

        for node_a, node_b, _ in self.edges:
            xa, ya = self._positions[node_a]
            xb, yb = self._positions[node_b]

            # Compute projection onto line segment [a, b]
            dx = xb - xa
            dy = yb - ya
            length_sq = dx * dx + dy * dy

            if length_sq == 0:
                # Degenerate edge (both nodes at same position)
                proj_x, proj_y = xa, ya
            else:
                # Parameter t for projection: 0 = at node_a, 1 = at node_b
                t = max(0.0, min(1.0, ((x - xa) * dx + (y - ya) * dy) / length_sq))
                proj_x = xa + t * dx
                proj_y = ya + t * dy

            dist = math.hypot(x - proj_x, y - proj_y)

            if dist < nearest_dist:
                if max_radius is None or dist <= max_radius:
                    nearest_dist = dist
                    nearest_projection = ((proj_x, proj_y), node_a, node_b, dist)

        return nearest_projection
