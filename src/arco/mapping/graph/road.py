"""RoadGraph: weighted graph with per-edge geometry metadata for road networks."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .weighted import WeightedGraph


class RoadGraph(WeightedGraph):
    """Weighted graph extended with per-edge geometry metadata.

    In addition to the standard WeightedGraph functionality, this class stores
    sequential waypoints along each edge. These waypoints can be used for
    spline interpolation, path smoothing, or trajectory generation.

    Each edge between nodes A and B stores an ordered list of intermediate
    (x, y) waypoints that define the road's geometry. The waypoints do not
    include the start and end nodes themselves.
    """

    def __init__(self) -> None:
        """Initialize an empty road graph."""
        super().__init__()
        # Map from edge (min_id, max_id) to list of waypoints
        self._edge_geometry: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}

    def add_edge(
        self,
        node_a: int,
        node_b: int,
        weight: Optional[float] = None,
        waypoints: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        """Add an undirected weighted edge with optional geometry waypoints.

        Args:
            node_a: ID of the first node.
            node_b: ID of the second node.
            weight: Edge weight. Defaults to the Euclidean distance between
                the two node positions when *None*.
            waypoints: Optional list of (x, y) intermediate waypoints along
                the edge. These points define the road geometry between the
                two nodes and can be used for spline interpolation.
        """
        super().add_edge(node_a, node_b, weight)

        # Store geometry in canonical order (min_id, max_id)
        edge_key = (min(node_a, node_b), max(node_a, node_b))
        if waypoints is not None:
            self._edge_geometry[edge_key] = list(waypoints)
        else:
            self._edge_geometry[edge_key] = []

    def edge_geometry(self, node_a: int, node_b: int) -> List[Tuple[float, float]]:
        """Return the waypoints defining the geometry of an edge.

        Args:
            node_a: ID of the first node.
            node_b: ID of the second node.

        Returns:
            List of (x, y) waypoints. Empty list if no waypoints were specified
            for this edge.
        """
        edge_key = (min(node_a, node_b), max(node_a, node_b))
        return self._edge_geometry.get(edge_key, [])

    def full_edge_geometry(self, node_a: int, node_b: int) -> List[Tuple[float, float]]:
        """Return the complete edge geometry including start and end nodes.

        Args:
            node_a: ID of the first node.
            node_b: ID of the second node.

        Returns:
            List of (x, y) points starting at node_a, including all intermediate
            waypoints, and ending at node_b.
        """
        waypoints = self.edge_geometry(node_a, node_b)
        edge_key = (min(node_a, node_b), max(node_a, node_b))

        # Determine traversal direction
        if node_a == edge_key[0]:
            # Forward direction: a -> waypoints -> b
            return [self.position(node_a)] + waypoints + [self.position(node_b)]
        else:
            # Reverse direction: a -> reverse(waypoints) -> b
            return [self.position(node_a)] + waypoints[::-1] + [self.position(node_b)]
