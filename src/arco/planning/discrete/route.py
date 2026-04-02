"""RouteRouter: route planning with continuous coordinate projection."""

from __future__ import annotations

import logging
import math
from typing import List, NamedTuple, Optional, Tuple

from ...mapping.graph.weighted import WeightedGraph
from .astar import AStarPlanner

logger = logging.getLogger(__name__)


class RouteResult(NamedTuple):
    """Result of a route planning query.

    Attributes:
        path: Sequence of node IDs from start to goal (inclusive).
        start_node: ID of the node nearest to the start position.
        goal_node: ID of the node nearest to the goal position.
        start_projection: (x, y) coordinates of the projected start position.
        goal_projection: (x, y) coordinates of the projected goal position.
        start_distance: Distance from start position to projected point.
        goal_distance: Distance from goal position to projected point.
    """

    path: List[int]
    start_node: int
    goal_node: int
    start_projection: Tuple[float, float]
    goal_projection: Tuple[float, float]
    start_distance: float
    goal_distance: float


class RouteRouter:
    """Route planner for continuous coordinates on weighted road graphs.

    Projects continuous (x, y) start/goal positions onto the nearest graph
    nodes, then runs A* to find the optimal path. Designed for autonomous
    navigation scenarios where agents have continuous positions but must
    follow discrete road networks.

    Example:
        >>> graph = WeightedGraph()
        >>> # Add nodes and edges...
        >>> router = RouteRouter(graph, activation_radius=50.0)
        >>> result = router.plan(start_x=10.5, start_y=20.3,
        ...                       goal_x=100.7, goal_y=200.9)
        >>> if result is not None:
        ...     print(f"Path: {result.path}")
        ...     print(f"Start node: {result.start_node}")
    """

    def __init__(
        self, graph: WeightedGraph, activation_radius: Optional[float] = None
    ) -> None:
        """Initialize RouteRouter.

        Args:
            graph: The weighted road graph to plan on.
            activation_radius: Maximum distance for projecting start/goal
                positions onto graph nodes. If None, any distance is accepted.
                Recommended to set this to prevent routing from positions far
                from valid roads.
        """
        self.graph = graph
        self.activation_radius = activation_radius
        self._planner = AStarPlanner(graph)

    def plan(
        self, start_x: float, start_y: float, goal_x: float, goal_y: float
    ) -> Optional[RouteResult]:
        """Plan a route from continuous start to continuous goal.

        Projects the start and goal positions onto the nearest graph nodes
        within the activation radius, then computes the shortest path using A*.

        Args:
            start_x: X coordinate of the start position.
            start_y: Y coordinate of the start position.
            goal_x: X coordinate of the goal position.
            goal_y: Y coordinate of the goal position.

        Returns:
            RouteResult with path and projection metadata, or None if:
                - Start position is outside activation radius
                - Goal position is outside activation radius
                - No path exists between projected nodes (disconnected graph)
        """
        logger.debug(
            "RouteRouter.plan: start=(%.2f, %.2f) goal=(%.2f, %.2f)",
            start_x,
            start_y,
            goal_x,
            goal_y,
        )
        # Project start position to nearest node
        start_node = self.graph.find_nearest_node(
            start_x, start_y, self.activation_radius
        )
        if start_node is None:
            logger.debug("RouteRouter: start outside activation radius")
            return None  # Start outside activation radius

        # Project goal position to nearest node
        goal_node = self.graph.find_nearest_node(
            goal_x, goal_y, self.activation_radius
        )
        if goal_node is None:
            logger.debug("RouteRouter: goal outside activation radius")
            return None  # Goal outside activation radius

        # Compute start/goal projection coordinates and distances
        start_proj_x, start_proj_y = self.graph.position(start_node)
        goal_proj_x, goal_proj_y = self.graph.position(goal_node)

        start_distance = math.hypot(
            start_x - start_proj_x, start_y - start_proj_y
        )
        goal_distance = math.hypot(goal_x - goal_proj_x, goal_y - goal_proj_y)

        # Run A* on the graph
        path = self._planner.plan(start_node, goal_node)
        if path is None:
            logger.debug(
                "RouteRouter: no path from node %s to node %s",
                start_node,
                goal_node,
            )
            return None  # No path exists (disconnected graph)

        logger.debug(
            "RouteRouter: path found (%d nodes, %s \u2192 %s)",
            len(path),
            start_node,
            goal_node,
        )
        return RouteResult(
            path=path,
            start_node=start_node,
            goal_node=goal_node,
            start_projection=(start_proj_x, start_proj_y),
            goal_projection=(goal_proj_x, goal_proj_y),
            start_distance=start_distance,
            goal_distance=goal_distance,
        )
