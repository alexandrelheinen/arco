"""RRTPlanner: Asymptotically-optimal RRT* for geometric planning."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from arco.mapping.occupancy import Occupancy

from .base import ContinuousPlanner

logger = logging.getLogger(__name__)


class RRTPlanner(ContinuousPlanner):
    """Asymptotically-optimal RRT* planner for continuous geometric spaces.

    Grows a randomised tree from *start*, rewiring existing edges to
    minimize path cost whenever a newly sampled node can improve the
    cost-to-come of nearby nodes.  As the sample count grows, the
    solution cost converges to the global optimum (Karaman & Frazzoli,
    2011).

    Collision checking is delegated to the :class:`~arco.mapping.Occupancy`
    interface: a straight-line segment is discretized into
    *collision_check_count* intermediate points and each is tested with
    :meth:`~arco.mapping.Occupancy.is_occupied`.

    Args:
        occupancy: The occupancy map used for collision checking.
        bounds: Axis-aligned bounding box for random sampling given as a
            sequence of ``(min, max)`` pairs — one per spatial dimension.
        max_sample_count: Maximum number of iterations (tree extensions).
        step_size: Maximum extension length per iteration (world units).
        goal_tolerance: Distance threshold within which a node is
            considered to have reached the goal.
        rewire_radius: Search radius for RRT* rewiring.  When ``None`` the
            radius shrinks as ``gamma * (log(n)/n)^(1/d)`` where *n* is
            the current tree size, *d* the dimension, and *gamma* is
            computed from the sampling volume to guarantee asymptotic
            optimality.
        collision_check_count: Number of intermediate points used when
            checking a new edge for collisions.
        goal_bias: Probability of sampling the goal state directly instead
            of a uniform random state.  Small values (0.05–0.1) speed up
            convergence without sacrificing coverage.
        early_stop: If ``True``, terminate as soon as the first node within
            *goal_tolerance* is found.  Set to ``False`` to keep iterating
            for a lower-cost solution (asymptotic optimality mode).
    """

    def __init__(
        self,
        occupancy: Occupancy,
        bounds: Sequence[Tuple[float, float]],
        max_sample_count: int = 2000,
        step_size: float = 1.0,
        goal_tolerance: float = 1.0,
        rewire_radius: Optional[float] = None,
        collision_check_count: int = 10,
        goal_bias: float = 0.05,
        early_stop: bool = True,
    ) -> None:
        """Initialize RRTPlanner.

        Args:
            occupancy: The occupancy map for collision checking.
            bounds: Sampling bounds as ``[(min_0, max_0), …, (min_d, max_d)]``.
            max_sample_count: Maximum number of iterations.
            step_size: Maximum extension step size (world units).
            goal_tolerance: Distance to goal that triggers path extraction.
            rewire_radius: Fixed rewire radius.  Adaptive when ``None``.
            collision_check_count: Segment resolution for collision checks.
            goal_bias: Probability of sampling the goal directly.
            early_stop: If ``True``, stop at the first node that reaches the
                goal.  If ``False``, run all iterations to optimize cost.

        Raises:
            ValueError: If *bounds* is empty or *step_size* is not positive.
        """
        super().__init__(occupancy)
        if not bounds:
            raise ValueError("bounds must not be empty.")
        if step_size <= 0:
            raise ValueError(f"step_size must be positive, got {step_size!r}.")
        self.bounds = list(bounds)
        self.max_sample_count = max_sample_count
        self.step_size = step_size
        self.goal_tolerance = goal_tolerance
        self._fixed_rewire_radius = rewire_radius
        self.collision_check_count = collision_check_count
        self.goal_bias = goal_bias
        self.early_stop = early_stop
        self._dim = len(bounds)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
    ) -> Optional[List[np.ndarray]]:
        """Plan a collision-free path from *start* to *goal* using RRT*.

        Args:
            start: Start position as a numpy array of shape ``(D,)``.
            goal: Goal position as a numpy array of shape ``(D,)``.

        Returns:
            An ordered list of numpy arrays ``[start, …, goal]`` or
            ``None`` if no path was found within *max_sample_count*.
        """
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)

        if self.occupancy.is_occupied(start):
            logger.debug("RRT*: start is occupied — no path.")
            return None
        if self.occupancy.is_occupied(goal):
            logger.debug("RRT*: goal is occupied — no path.")
            return None

        # Tree stored as parallel arrays for efficient nearest-node search.
        nodes: List[np.ndarray] = [start]
        parent: Dict[int, Optional[int]] = {0: None}
        cost: Dict[int, float] = {0: 0.0}

        best_goal_node: Optional[int] = None
        best_goal_cost = math.inf

        rng = np.random.default_rng()

        for iteration in range(self.max_sample_count):
            # --- Sample ---------------------------------------------------
            if rng.random() < self.goal_bias:
                x_rand = goal.copy()
            else:
                x_rand = self._sample(rng)

            # --- Nearest --------------------------------------------------
            nearest_idx = self._nearest(nodes, x_rand)
            x_nearest = nodes[nearest_idx]

            # --- Steer ----------------------------------------------------
            x_new = self._steer(x_nearest, x_rand)

            # --- Collision check on edge ---------------------------------
            if not self._segment_free(x_nearest, x_new):
                continue

            # --- Near nodes -----------------------------------------------
            node_count = len(nodes)
            radius = self._rewire_radius(node_count)
            near_idxs = self._near(nodes, x_new, radius)

            # --- Choose best parent --------------------------------------
            best_parent_idx = nearest_idx
            best_cost = cost[nearest_idx] + float(
                np.linalg.norm(x_new - x_nearest)
            )
            for idx in near_idxs:
                if idx == nearest_idx:
                    continue
                c = cost[idx] + float(np.linalg.norm(x_new - nodes[idx]))
                if c < best_cost and self._segment_free(nodes[idx], x_new):
                    best_cost = c
                    best_parent_idx = idx

            # --- Add node ------------------------------------------------
            new_idx = node_count
            nodes.append(x_new)
            parent[new_idx] = best_parent_idx
            cost[new_idx] = best_cost

            # --- Rewire --------------------------------------------------
            for idx in near_idxs:
                if idx == best_parent_idx:
                    continue
                c_through_new = best_cost + float(
                    np.linalg.norm(nodes[idx] - x_new)
                )
                if c_through_new < cost[idx] and self._segment_free(
                    x_new, nodes[idx]
                ):
                    parent[idx] = new_idx
                    cost[idx] = c_through_new

            # --- Goal check -----------------------------------------------
            dist_to_goal = float(np.linalg.norm(x_new - goal))
            if (
                dist_to_goal <= self.goal_tolerance
                and best_cost < best_goal_cost
            ):
                best_goal_cost = best_cost
                best_goal_node = new_idx
                logger.debug(
                    "RRT*: reached goal at iter %d, cost=%.3f",
                    iteration,
                    best_cost,
                )
                if self.early_stop:
                    break

        if best_goal_node is None:
            logger.debug("RRT*: no path found.")
            return None

        path = self._extract_path(nodes, parent, best_goal_node)
        path.append(goal.copy())
        logger.debug(
            "RRT*: path extracted, %d waypoints, cost=%.3f",
            len(path),
            best_goal_cost,
        )
        return path

    # ------------------------------------------------------------------
    # Tree helpers (also useful for external visualisation)
    # ------------------------------------------------------------------

    def get_tree(
        self,
        start: np.ndarray,
        goal: np.ndarray,
    ) -> Tuple[
        List[np.ndarray],
        Dict[int, Optional[int]],
        Optional[List[np.ndarray]],
    ]:
        """Run RRT* and return the full exploration tree alongside the path.

        Useful for visualisation tools that want to render the tree edges.

        Args:
            start: Start position as a numpy array.
            goal: Goal position as a numpy array.

        Returns:
            A ``(nodes, parent, path)`` tuple where *nodes* is the list of
            all tree nodes, *parent* maps each node index to its parent
            index (``None`` for the root), and *path* is the solution path
            or ``None`` if no path was found.
        """
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)

        if self.occupancy.is_occupied(start) or self.occupancy.is_occupied(
            goal
        ):
            return [start], {0: None}, None

        nodes: List[np.ndarray] = [start]
        parent: Dict[int, Optional[int]] = {0: None}
        cost: Dict[int, float] = {0: 0.0}

        best_goal_node: Optional[int] = None
        best_goal_cost = math.inf

        rng = np.random.default_rng()

        for _ in range(self.max_sample_count):
            if rng.random() < self.goal_bias:
                x_rand = goal.copy()
            else:
                x_rand = self._sample(rng)

            nearest_idx = self._nearest(nodes, x_rand)
            x_nearest = nodes[nearest_idx]
            x_new = self._steer(x_nearest, x_rand)

            if not self._segment_free(x_nearest, x_new):
                continue

            node_count = len(nodes)
            radius = self._rewire_radius(node_count)
            near_idxs = self._near(nodes, x_new, radius)

            best_parent_idx = nearest_idx
            best_cost = cost[nearest_idx] + float(
                np.linalg.norm(x_new - x_nearest)
            )
            for idx in near_idxs:
                if idx == nearest_idx:
                    continue
                c = cost[idx] + float(np.linalg.norm(x_new - nodes[idx]))
                if c < best_cost and self._segment_free(nodes[idx], x_new):
                    best_cost = c
                    best_parent_idx = idx

            new_idx = node_count
            nodes.append(x_new)
            parent[new_idx] = best_parent_idx
            cost[new_idx] = best_cost

            for idx in near_idxs:
                if idx == best_parent_idx:
                    continue
                c_through_new = best_cost + float(
                    np.linalg.norm(nodes[idx] - x_new)
                )
                if c_through_new < cost[idx] and self._segment_free(
                    x_new, nodes[idx]
                ):
                    parent[idx] = new_idx
                    cost[idx] = c_through_new

            dist_to_goal = float(np.linalg.norm(x_new - goal))
            if (
                dist_to_goal <= self.goal_tolerance
                and best_cost < best_goal_cost
            ):
                best_goal_cost = best_cost
                best_goal_node = new_idx
                if self.early_stop:
                    break

        if best_goal_node is None:
            return nodes, parent, None

        path = self._extract_path(nodes, parent, best_goal_node)
        path.append(goal.copy())
        return nodes, parent, path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample(self, rng: np.random.Generator) -> np.ndarray:
        """Sample a random point uniformly within :attr:`bounds`.

        Args:
            rng: NumPy random generator to use.

        Returns:
            A random position as a numpy array of shape ``(D,)``.
        """
        lo = np.array([b[0] for b in self.bounds])
        hi = np.array([b[1] for b in self.bounds])
        return rng.uniform(lo, hi)

    def _nearest(self, nodes: List[np.ndarray], point: np.ndarray) -> int:
        """Return the index of the node in *nodes* closest to *point*.

        Args:
            nodes: Current tree nodes.
            point: Query point as a numpy array.

        Returns:
            Index of the nearest node in *nodes*.
        """
        dists = [float(np.linalg.norm(n - point)) for n in nodes]
        return int(np.argmin(dists))

    def _steer(self, from_pt: np.ndarray, to_pt: np.ndarray) -> np.ndarray:
        """Move from *from_pt* toward *to_pt* by at most :attr:`step_size`.

        Args:
            from_pt: Origin position.
            to_pt: Target position.

        Returns:
            New position at most :attr:`step_size` away from *from_pt*.
        """
        delta = to_pt - from_pt
        dist = float(np.linalg.norm(delta))
        if dist <= self.step_size:
            return to_pt.copy()
        return from_pt + (delta / dist) * self.step_size

    def _near(
        self, nodes: List[np.ndarray], point: np.ndarray, radius: float
    ) -> List[int]:
        """Return indices of all nodes within *radius* of *point*.

        Args:
            nodes: Current tree nodes.
            point: Query point.
            radius: Search radius.

        Returns:
            List of node indices within the radius.
        """
        return [
            i
            for i, n in enumerate(nodes)
            if float(np.linalg.norm(n - point)) <= radius
        ]

    def _rewire_radius(self, node_count: int) -> float:
        """Compute the adaptive RRT* rewire radius.

        Args:
            node_count: Current number of nodes in the tree.

        Returns:
            Rewire radius (world units).
        """
        if self._fixed_rewire_radius is not None:
            return self._fixed_rewire_radius
        if node_count <= 1:
            return self.step_size
        # gamma_star formula from Karaman & Frazzoli (2011)
        vol = math.prod(b[1] - b[0] for b in self.bounds)
        d = self._dim
        unit_ball_vol = (math.pi ** (d / 2.0)) / math.gamma(d / 2.0 + 1)
        gamma = (
            2.0
            * (1.0 + 1.0 / d) ** (1.0 / d)
            * (vol / unit_ball_vol) ** (1.0 / d)
        )
        return float(
            min(
                gamma * (math.log(node_count) / node_count) ** (1.0 / d),
                self.step_size * 2,
            )
        )

    def _segment_free(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Check whether the straight-line segment from *a* to *b* is free.

        Discretizes the segment into :attr:`collision_check_count`
        intermediate points and tests each with
        :meth:`~arco.mapping.Occupancy.is_occupied`.

        Args:
            a: Segment start position.
            b: Segment end position.

        Returns:
            True if no intermediate point is occupied, False otherwise.
        """
        for t in np.linspace(0.0, 1.0, self.collision_check_count + 2):
            pt = a + t * (b - a)
            if self.occupancy.is_occupied(pt):
                return False
        return True

    def _extract_path(
        self,
        nodes: List[np.ndarray],
        parent: Dict[int, Optional[int]],
        goal_node: int,
    ) -> List[np.ndarray]:
        """Trace back from *goal_node* to the root and return the path.

        Args:
            nodes: All tree nodes.
            parent: Parent index map.
            goal_node: Index of the goal-reaching node.

        Returns:
            Ordered list of waypoints from root to *goal_node*.
        """
        path: List[np.ndarray] = []
        idx: Optional[int] = goal_node
        while idx is not None:
            path.append(nodes[idx].copy())
            idx = parent[idx]
        path.reverse()
        return path
