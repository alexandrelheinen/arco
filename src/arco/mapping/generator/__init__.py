"""RoadNetworkGenerator: procedural generation of road networks with spline-aware edges."""

from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple

from ..graph.road import RoadGraph


class RoadNetworkGenerator:
    """Procedurally generates 2D road networks suitable for pathfinding and trajectory following.

    This generator creates road networks with:
    - Intersection nodes (road junctions)
    - Weighted edges connecting intersections
    - Geometry metadata (sequential waypoints) for each road segment
    - Deterministic generation based on random seed

    The generated graphs are compatible with AStarPlanner and include edge
    geometry that can be used for spline interpolation in path smoothing.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize the road network generator.

        Args:
            seed: Random seed for deterministic generation. If None, a random
                seed is chosen.
        """
        self._seed = seed if seed is not None else random.randint(0, 2**31 - 1)
        self._rng = random.Random(self._seed)

    @property
    def seed(self) -> int:
        """Return the random seed used for generation."""
        return self._seed

    def generate_grid_network(
        self,
        grid_size: Tuple[int, int] = (3, 3),
        cell_size: float = 100.0,
        waypoints_per_edge: int = 3,
        curvature: float = 0.2,
    ) -> RoadGraph:
        """Generate a grid-based road network with curved roads.

        Creates a regular grid of intersections connected by roads. Each road
        segment has intermediate waypoints that define its geometry, allowing
        for curved roads between straight intersections.

        Args:
            grid_size: Number of intersections in (rows, cols) format.
            cell_size: Distance between adjacent intersections.
            waypoints_per_edge: Number of intermediate waypoints per road segment.
            curvature: Maximum perpendicular deviation of waypoints from the
                straight line between intersections (as fraction of edge length).
                0.0 = straight roads, 1.0 = highly curved roads.

        Returns:
            A RoadGraph with nodes at grid intersections and edges with
            geometry waypoints.

        Raises:
            ValueError: If grid_size has non-positive dimensions.
        """
        rows, cols = grid_size
        if rows <= 0 or cols <= 0:
            raise ValueError("Grid dimensions must be positive")

        graph = RoadGraph()

        # Create intersection nodes
        node_id = 0
        node_positions: List[Tuple[int, int, int]] = []  # (node_id, row, col)
        for row in range(rows):
            for col in range(cols):
                x = col * cell_size
                y = row * cell_size
                graph.add_node(node_id, x, y)
                node_positions.append((node_id, row, col))
                node_id += 1

        # Create road segments (edges) with geometry
        for nid, row, col in node_positions:
            # Horizontal edge (right)
            if col < cols - 1:
                right_nid = nid + 1
                waypoints = self._generate_waypoints(
                    graph.position(nid),
                    graph.position(right_nid),
                    waypoints_per_edge,
                    curvature,
                )
                graph.add_edge(nid, right_nid, waypoints=waypoints)

            # Vertical edge (down)
            if row < rows - 1:
                down_nid = nid + cols
                waypoints = self._generate_waypoints(
                    graph.position(nid),
                    graph.position(down_nid),
                    waypoints_per_edge,
                    curvature,
                )
                graph.add_edge(nid, down_nid, waypoints=waypoints)

        return graph

    def generate_random_network(
        self,
        num_intersections: int = 20,
        area: float = 500.0,
        connect_radius: float = 150.0,
        waypoints_per_edge: int = 3,
        curvature: float = 0.15,
    ) -> RoadGraph:
        """Generate a random road network with curved roads.

        Creates randomly positioned intersections and connects nearby ones
        with curved road segments. This produces more organic, less regular
        road networks.

        Args:
            num_intersections: Number of intersection nodes to generate.
            area: Side length of the square area for placing intersections.
            connect_radius: Maximum distance for connecting two intersections.
            waypoints_per_edge: Number of intermediate waypoints per road segment.
            curvature: Maximum perpendicular deviation of waypoints from the
                straight line between intersections (as fraction of edge length).

        Returns:
            A RoadGraph with randomly placed intersections and curved road
            segments.

        Raises:
            ValueError: If num_intersections is non-positive.
        """
        if num_intersections <= 0:
            raise ValueError("Number of intersections must be positive")

        graph = RoadGraph()

        # Create randomly placed intersections
        for i in range(num_intersections):
            x = self._rng.uniform(0.0, area)
            y = self._rng.uniform(0.0, area)
            graph.add_node(i, x, y)

        # Connect nearby intersections with curved roads
        for i in range(num_intersections):
            for j in range(i + 1, num_intersections):
                xi, yi = graph.position(i)
                xj, yj = graph.position(j)
                distance = math.hypot(xi - xj, yi - yj)

                if distance <= connect_radius:
                    waypoints = self._generate_waypoints(
                        (xi, yi),
                        (xj, yj),
                        waypoints_per_edge,
                        curvature,
                    )
                    graph.add_edge(i, j, waypoints=waypoints)

        return graph

    def _generate_waypoints(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        num_waypoints: int,
        curvature: float,
    ) -> List[Tuple[float, float]]:
        """Generate intermediate waypoints along an edge with controlled curvature.

        Args:
            start: (x, y) position of the start node.
            end: (x, y) position of the end node.
            num_waypoints: Number of intermediate waypoints to generate.
            curvature: Maximum perpendicular deviation from straight line
                (as fraction of edge length).

        Returns:
            List of (x, y) waypoints between start and end (not including
            start and end themselves).
        """
        if num_waypoints <= 0:
            return []

        waypoints: List[Tuple[float, float]] = []
        sx, sy = start
        ex, ey = end

        # Calculate edge vector and perpendicular
        dx = ex - sx
        dy = ey - sy
        length = math.hypot(dx, dy)

        if length == 0:
            # Start and end are the same, no waypoints
            return []

        # Unit vectors
        ux = dx / length
        uy = dy / length
        perp_x = -uy  # Perpendicular unit vector
        perp_y = ux

        # Maximum perpendicular deviation
        max_deviation = length * curvature

        for i in range(1, num_waypoints + 1):
            # Position along the line (evenly spaced)
            t = i / (num_waypoints + 1)
            base_x = sx + t * dx
            base_y = sy + t * dy

            # Add random perpendicular offset
            # Use sine wave pattern for smooth curves with random phase and amplitude
            phase = self._rng.uniform(0, 2 * math.pi)
            amplitude = self._rng.uniform(-max_deviation, max_deviation)
            offset = amplitude * math.sin(t * math.pi + phase)

            waypoint_x = base_x + offset * perp_x
            waypoint_y = base_y + offset * perp_y

            waypoints.append((waypoint_x, waypoint_y))

        return waypoints
