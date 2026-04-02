"""RouteRouter: route planning with N-dimensional coordinate projection."""

from __future__ import annotations

import logging
from typing import List, NamedTuple, Optional

import numpy as np

from ...mapping.graph.cartesian import CartesianGraph
from .astar import AStarPlanner

logger = logging.getLogger(__name__)


class RouteResult(NamedTuple):
    """Result of a route planning query.

    Attributes:
        path: Sequence of node IDs from start to goal (inclusive).
        start_node: ID of the node nearest to the start position.
        goal_node: ID of the node nearest to the goal position.
        start_projection: Cartesian position of the projected start node
            as a :class:`numpy.ndarray`.
        goal_projection: Cartesian position of the projected goal node
            as a :class:`numpy.ndarray`.
        start_distance: Distance from start position to projected point.
        goal_distance: Distance from goal position to projected point.
    """

    path: List[int]
    start_node: int
    goal_node: int
    start_projection: np.ndarray
    goal_projection: np.ndarray
    start_distance: float
    goal_distance: float


class RouteRouter:
    """Route planner for continuous coordinates on Cartesian road graphs.

    Projects continuous start/goal positions onto the nearest graph nodes,
    then runs A* to find the optimal path.  Works for any spatial dimension
    N.  Designed for autonomous navigation scenarios where agents have
    continuous positions but must follow discrete road networks.

    Example:
        >>> import numpy as np
        >>> graph = CartesianGraph()
        >>> # Add nodes and edges...
        >>> router = RouteRouter(graph, activation_radius=50.0)
        >>> result = router.plan(
        ...     start_position=np.array([10.5, 20.3]),
        ...     goal_position=np.array([100.7, 200.9]),
        ... )
        >>> if result is not None:
        ...     print(f"Path: {result.path}")
        ...     print(f"Start node: {result.start_node}")
    """

    def __init__(
        self,
        graph: CartesianGraph,
        activation_radius: Optional[float] = None,
    ) -> None:
        """Initialize RouteRouter.

        Args:
            graph: The Cartesian road graph to plan on.
            activation_radius: Maximum distance for projecting start/goal
                positions onto graph nodes.  If ``None``, any distance is
                accepted.  Recommended to set this to prevent routing from
                positions far from valid roads.
        """
        self.graph = graph
        self.activation_radius = activation_radius
        self._planner = AStarPlanner(graph)

    def plan(
        self,
        start_position: np.ndarray,
        goal_position: np.ndarray,
    ) -> Optional[RouteResult]:
        """Plan a route from continuous start to continuous goal.

        Projects the start and goal positions onto the nearest graph nodes
        within the activation radius, then computes the shortest path
        using A*.

        Args:
            start_position: Cartesian position of the start as a
                :class:`numpy.ndarray` of shape ``(N,)``.
            goal_position: Cartesian position of the goal as a
                :class:`numpy.ndarray` of shape ``(N,)``.

        Returns:
            :class:`RouteResult` with path and projection metadata, or
            ``None`` if:

            - Start position is outside the activation radius.
            - Goal position is outside the activation radius.
            - No path exists between the projected nodes.
        """
        start_position = np.asarray(start_position, dtype=float)
        goal_position = np.asarray(goal_position, dtype=float)

        logger.debug(
            "RouteRouter.plan: start=%s goal=%s",
            start_position,
            goal_position,
        )

        start_node = self.graph.find_nearest_node(
            start_position, self.activation_radius
        )
        if start_node is None:
            logger.debug("RouteRouter: start outside activation radius")
            return None

        goal_node = self.graph.find_nearest_node(
            goal_position, self.activation_radius
        )
        if goal_node is None:
            logger.debug("RouteRouter: goal outside activation radius")
            return None

        start_proj = self.graph.position(start_node)
        goal_proj = self.graph.position(goal_node)

        start_distance = float(np.linalg.norm(start_position - start_proj))
        goal_distance = float(np.linalg.norm(goal_position - goal_proj))

        path = self._planner.plan(start_node, goal_node)
        if path is None:
            logger.debug(
                "RouteRouter: no path from node %s to node %s",
                start_node,
                goal_node,
            )
            return None

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
            start_projection=start_proj,
            goal_projection=goal_proj,
            start_distance=start_distance,
            goal_distance=goal_distance,
        )
