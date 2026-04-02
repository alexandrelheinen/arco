"""CartesianGraph: weighted graph with N-dimensional Cartesian node positions."""

# Copyright 2026 Alexandre Loeblein Heinen

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from .weighted import WeightedGraph


class CartesianGraph(WeightedGraph):
    """Weighted graph whose nodes carry N-dimensional Cartesian positions.

    Nodes are identified by integer IDs and store their position as a
    :class:`numpy.ndarray` of shape ``(N,)``.  Edge weight defaults to the
    Euclidean distance between the two endpoint positions; override
    :meth:`_compute_distance` to change the metric.

    Works for any dimension N: pass 2 coordinates for a 2-D graph, 3 for
    3-D, etc.

    Implements the same ``neighbors`` / ``distance`` / ``heuristic``
    interface as :class:`~arco.mapping.grid.base.Grid` so that
    :class:`~arco.planning.discrete.astar.AStarPlanner` can operate on it
    without modification.
    """

    def __init__(self, ndim: Optional[int] = None) -> None:
        """Initialize an empty Cartesian graph.

        Args:
            ndim: Expected number of spatial dimensions.  When set, every
                subsequent call to :meth:`add_node` validates that the
                supplied position has exactly this many coordinates.
                Pass ``None`` (the default) to infer the dimension from the
                first node added.
        """
        super().__init__()
        self._positions: Dict[int, np.ndarray] = {}
        self._ndim: Optional[int] = ndim

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------

    def add_node(self, node_id: int, *coords: float) -> None:
        """Add a node with a Cartesian position.

        Args:
            node_id: Unique integer identifier for the node.
            *coords: Coordinate values that define the node position.
                For a 2-D graph pass two floats ``(x, y)``; for 3-D pass
                three floats ``(x, y, z)``, and so on.

        Raises:
            ValueError: If *coords* is empty or its length does not match
                the dimension of previously added nodes.
        """
        if not coords:
            raise ValueError(
                "At least one coordinate is required to add a node."
            )
        position = np.array(coords, dtype=float)
        if self._ndim is None:
            self._ndim = len(position)
        elif len(position) != self._ndim:
            raise ValueError(
                f"Position dimension mismatch: expected {self._ndim},"
                f" got {len(position)}."
            )
        self._positions[node_id] = position
        super().add_node(node_id)

    def position(self, node_id: int) -> np.ndarray:
        """Return the Cartesian position of a node.

        Args:
            node_id: ID of the node.

        Returns:
            Position as a :class:`numpy.ndarray` of shape ``(N,)``.
        """
        return self._positions[node_id]

    @property
    def ndim(self) -> Optional[int]:
        """Number of spatial dimensions inferred from added nodes."""
        return self._ndim

    # ------------------------------------------------------------------
    # Edge management
    # ------------------------------------------------------------------

    def add_edge(
        self, node_a: int, node_b: int, weight: Optional[float] = None
    ) -> None:
        """Add an undirected weighted edge between two nodes.

        Args:
            node_a: ID of the first node.
            node_b: ID of the second node.
            weight: Edge weight.  Defaults to the Euclidean distance
                between the two node positions when *None*.
        """
        if weight is None:
            weight = self._euclidean(node_a, node_b)
        super().add_edge(node_a, node_b, weight)

    # ------------------------------------------------------------------
    # Distance and heuristic
    # ------------------------------------------------------------------

    def _compute_distance(self, pos_a: np.ndarray, pos_b: np.ndarray) -> float:
        """Return the distance between two position vectors.

        Subclasses can override this method to use a different metric
        (e.g. Manhattan distance for a grid-like graph).  The default
        implementation returns the Euclidean (L2) distance.

        Args:
            pos_a: Position of the first node.
            pos_b: Position of the second node.

        Returns:
            Distance as a float.
        """
        return float(np.linalg.norm(pos_a - pos_b))

    def _euclidean(self, node_a: int, node_b: int) -> float:
        return self._compute_distance(
            self._positions[node_a], self._positions[node_b]
        )

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

    def heuristic(self, node_a: int, node_b: int) -> float:
        """Return the Euclidean distance between two nodes.

        This admissible, consistent heuristic is suitable for A* search
        on any Cartesian graph regardless of dimension.

        Args:
            node_a: ID of the first node.
            node_b: ID of the second node.

        Returns:
            Euclidean distance as a float.
        """
        return self._euclidean(node_a, node_b)

    # ------------------------------------------------------------------
    # Spatial queries
    # ------------------------------------------------------------------

    def find_nearest_node(
        self,
        position: np.ndarray,
        max_radius: Optional[float] = None,
    ) -> Optional[int]:
        """Find the node closest to the given N-D position.

        Args:
            position: Query position as a :class:`numpy.ndarray` of
                shape ``(N,)``.
            max_radius: Maximum search radius.  When set, only nodes
                within this distance are considered; ``None`` is returned
                if no node falls within the radius.

        Returns:
            ID of the nearest node, or ``None`` if the graph is empty or
            no node lies within *max_radius*.
        """
        if not self._positions:
            return None

        position = np.asarray(position, dtype=float)
        nearest_node: Optional[int] = None
        nearest_dist = float("inf")

        for node_id, node_pos in self._positions.items():
            dist = self._compute_distance(position, node_pos)
            if dist < nearest_dist:
                if max_radius is None or dist <= max_radius:
                    nearest_dist = dist
                    nearest_node = node_id

        return nearest_node

    def project_to_nearest_edge(
        self,
        position: np.ndarray,
        max_radius: Optional[float] = None,
    ) -> Optional[Tuple[np.ndarray, int, int, float]]:
        """Project a point onto the nearest edge of the graph.

        Finds the closest point on any edge to the query position by
        computing the perpendicular projection onto each edge line segment.
        Works for any spatial dimension N.

        Args:
            position: Query position as a :class:`numpy.ndarray` of
                shape ``(N,)``.
            max_radius: Maximum search radius.  When set, only edges
                whose projection is within this distance are considered.

        Returns:
            A tuple containing:

            - ``proj``: The projected point as a :class:`numpy.ndarray`.
            - ``node_a``: ID of the first endpoint of the nearest edge.
            - ``node_b``: ID of the second endpoint of the nearest edge.
            - ``distance``: Distance from the query point to the
              projection.

            Returns ``None`` if no edges exist or none fall within
            *max_radius*.
        """
        if not self.edges:
            return None

        position = np.asarray(position, dtype=float)
        nearest_projection: Optional[Tuple[np.ndarray, int, int, float]] = None
        nearest_dist = float("inf")

        for node_a, node_b, _ in self.edges:
            pos_a = self._positions[node_a]
            pos_b = self._positions[node_b]

            edge_vec = pos_b - pos_a
            length_sq = float(np.dot(edge_vec, edge_vec))

            if length_sq == 0.0:
                proj = pos_a.copy()
            else:
                t = float(np.dot(position - pos_a, edge_vec) / length_sq)
                t = max(0.0, min(1.0, t))
                proj = pos_a + t * edge_vec

            dist = self._compute_distance(position, proj)

            if dist < nearest_dist:
                if max_radius is None or dist <= max_radius:
                    nearest_dist = dist
                    nearest_projection = (proj, node_a, node_b, dist)

        return nearest_projection
