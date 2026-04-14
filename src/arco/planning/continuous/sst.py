"""SSTPlanner: Sparse Stable Trees planner for geometric planning."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from arco.mapping.occupancy import Occupancy

from .base import ContinuousPlanner

logger = logging.getLogger(__name__)


class SSTPlanner(ContinuousPlanner):
    """Asymptotically near-optimal SST planner for continuous geometric spaces.

    SST (Stable Sparse Trees, Li et al. 2016) maintains a *sparse*
    representative subset of tree nodes to balance exploration quality and
    memory.  For each *witness* cell (a δ_s-spaced virtual grid), only the
    active node with the lowest cost-to-come is kept as the cell's
    representative.  When a cheaper representative is found the old one is
    removed, keeping the tree compact.

    Because ARCO targets purely *geometric* (no-dynamics) planning here,
    the propagation step steers the nearest active node toward the random
    sample by at most *step_size* — the same as RRT*.

    Collision checking is done the same way as in
    :class:`~arco.planning.continuous.rrt.RRTPlanner`.

    Args:
        occupancy: The occupancy map used for collision checking.
        bounds: Axis-aligned sampling bounds as ``[(min, max), …]``.
        max_sample_count: Maximum number of propagation attempts.
        step_size: Per-dimension maximum extension as a scalar or 1-D
            array of shape ``(D,)``.  All distances (nearest-neighbor
            selection, witness proximity, goal check) are measured in the
            space normalized by these scales.  A scalar is broadcast to
            all dimensions.
        goal_tolerance: Distance threshold in normalized units at which a
            node is considered to have reached the goal.
        witness_radius: Witness cell half-width (δ_s) in normalized units.
            Controls tree sparsity: smaller values → denser tree → better
            optimality.  Must be less than 1.0 (one normalized step) to
            allow tree growth.
        collision_check_count: Segment resolution for collision checks.
        goal_bias: Probability of sampling the goal state directly.
        early_stop: If ``True``, terminate as soon as the first node within
            *goal_tolerance* is found.  Set to ``False`` to keep iterating
            for a lower-cost solution (asymptotic optimality mode).
    """

    def __init__(
        self,
        occupancy: Occupancy,
        bounds: Sequence[Tuple[float, float]],
        max_sample_count: int = 3000,
        step_size: float | np.ndarray = 1.0,
        goal_tolerance: float = 1.0,
        witness_radius: float = 0.5,
        collision_check_count: int = 10,
        goal_bias: float = 0.05,
        early_stop: bool = True,
    ) -> None:
        """Initialize SSTPlanner.

        Args:
            occupancy: The occupancy map for collision checking.
            bounds: Sampling bounds as ``[(min_0, max_0), …]``.
            max_sample_count: Maximum number of iterations.
            step_size: Per-dimension maximum extension as a scalar or 1-D
                array of shape ``(D,)``.  All distances are measured in the
                space normalized by these scales.
            goal_tolerance: Distance in normalized units that triggers path
                extraction.
            witness_radius: Half-width of witness cells (δ_s) in normalized
                units.  Should be less than 1.0 to allow tree growth.
            collision_check_count: Segment resolution for collision checks.
            goal_bias: Probability of sampling the goal directly.
            early_stop: If ``True``, stop at the first node that reaches the
                goal.  If ``False``, run all iterations to optimize cost.

        Raises:
            ValueError: If *bounds* is empty or any element of *step_size*
                is not positive.
        """
        super().__init__(occupancy)
        if not bounds:
            raise ValueError("bounds must not be empty.")
        self.step_size = np.asarray(step_size, dtype=float)
        if np.any(self.step_size <= 0):
            raise ValueError(f"step_size must be positive, got {step_size!r}.")
        self.bounds = list(bounds)
        self.max_sample_count = max_sample_count
        self.goal_tolerance = goal_tolerance
        self.witness_radius = witness_radius
        self.collision_check_count = collision_check_count
        self.goal_bias = goal_bias
        self.early_stop = early_stop

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
    ) -> Optional[List[np.ndarray]]:
        """Plan a collision-free path from *start* to *goal* using SST.

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
            logger.debug("SST: start is occupied — no path.")
            return None
        if self.occupancy.is_occupied(goal):
            logger.debug("SST: goal is occupied — no path.")
            return None

        _, _, path = self._run(start, goal)
        return path

    # ------------------------------------------------------------------
    # Tree helpers (for external visualisation)
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
        """Run SST and return the active tree alongside the solution path.

        Args:
            start: Start position as a numpy array.
            goal: Goal position as a numpy array.

        Returns:
            A ``(nodes, parent, path)`` tuple.  Only *active* nodes are
            included in *nodes*; *parent* maps each active node index to
            its parent index (``None`` for the root).  *path* is the
            solution path or ``None``.
        """
        return self._run(
            np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
        )

    # ------------------------------------------------------------------
    # Core SST algorithm
    # ------------------------------------------------------------------

    def _run(
        self,
        start: np.ndarray,
        goal: np.ndarray,
    ) -> Tuple[
        List[np.ndarray],
        Dict[int, Optional[int]],
        Optional[List[np.ndarray]],
    ]:
        """Run SST from *start* to *goal*.

        Args:
            start: Start position (already validated, numpy array).
            goal: Goal position (already validated, numpy array).

        Returns:
            ``(active_nodes, parent, path)`` — see :meth:`get_tree`.
        """
        nodes: List[np.ndarray] = [start]
        cost: Dict[int, float] = {0: 0.0}
        parent: Dict[int, Optional[int]] = {0: None}
        active: Set[int] = {0}
        # Witnesses: position + representative index
        witnesses: List[np.ndarray] = [start.copy()]
        witness_rep: List[Optional[int]] = [0]

        best_goal_node: Optional[int] = None
        best_goal_cost = math.inf
        rng = np.random.default_rng()

        for iteration in range(self.max_sample_count):
            # --- Sample --------------------------------------------------
            if rng.random() < self.goal_bias:
                x_rand = goal.copy()
            else:
                x_rand = self._sample(rng)

            # --- Select nearest active node to sample --------------------
            # Geometric SST uses nearest-neighbor selection, which allows
            # the tree to explore in the direction of the sample.
            x_selected_idx = self._select_active(nodes, active, x_rand)
            if x_selected_idx is None:
                continue
            x_selected = nodes[x_selected_idx]

            # --- Propagate -----------------------------------------------
            x_new = self._steer(x_selected, x_rand)

            # --- Collision check -----------------------------------------
            if not self._segment_free(x_selected, x_new):
                continue

            new_cost = cost[x_selected_idx] + float(
                np.linalg.norm((x_new - x_selected) / self.step_size)
            )

            # --- Witness update ------------------------------------------
            w_idx = self._nearest_witness(witnesses, x_new)
            if w_idx is None:
                # No witness nearby → create one for this region
                witnesses.append(x_new.copy())
                witness_rep.append(None)
                w_idx = len(witnesses) - 1

            rep_idx = witness_rep[w_idx]

            # Only add if this node is cheaper than the current rep
            if rep_idx is not None and new_cost >= cost.get(rep_idx, math.inf):
                continue

            # --- Add new node to tree ------------------------------------
            new_idx = len(nodes)
            nodes.append(x_new)
            cost[new_idx] = new_cost
            parent[new_idx] = x_selected_idx
            active.add(new_idx)

            # --- Remove dominated representative -------------------------
            if rep_idx is not None:
                active.discard(rep_idx)

            witness_rep[w_idx] = new_idx

            # --- Goal check ----------------------------------------------
            dist_to_goal = float(
                np.linalg.norm((x_new - goal) / self.step_size)
            )
            if (
                dist_to_goal <= self.goal_tolerance
                and new_cost < best_goal_cost
            ):
                best_goal_cost = new_cost
                best_goal_node = new_idx
                logger.debug(
                    "SST: reached goal at iter %d, cost=%.3f",
                    iteration,
                    new_cost,
                )
                if self.early_stop:
                    break

        if best_goal_node is None:
            logger.debug("SST: no path found.")
            active_nodes, active_parent = self._build_active_output(
                nodes, parent, active
            )
            return active_nodes, active_parent, None

        path = self._extract_path(nodes, parent, best_goal_node)
        # Only append the exact goal when the direct segment is free.
        # Large goal_tolerance values allow the last tree node to be far
        # enough from the goal that the connecting segment crosses an obstacle.
        if self._segment_free(nodes[best_goal_node], goal):
            path.append(goal.copy())
        active_nodes, active_parent = self._build_active_output(
            nodes, parent, active
        )
        logger.debug(
            "SST: path extracted, %d waypoints, cost=%.3f, tree size=%d",
            len(path),
            best_goal_cost,
            len(active),
        )
        return active_nodes, active_parent, path

    def _build_active_output(
        self,
        nodes: List[np.ndarray],
        parent: Dict[int, Optional[int]],
        active: Set[int],
    ) -> Tuple[List[np.ndarray], Dict[int, Optional[int]]]:
        """Build the re-indexed active-node list and parent map for output.

        Args:
            nodes: All tree nodes.
            parent: Parent map over all node indices.
            active: Set of active node indices.

        Returns:
            ``(active_nodes, active_parent)`` with indices in ``[0, …, N)``.
        """
        sorted_active = sorted(active)
        idx_map = {old: new for new, old in enumerate(sorted_active)}
        active_nodes = [nodes[i] for i in sorted_active]
        active_parent: Dict[int, Optional[int]] = {}
        for old_i in sorted_active:
            new_i = idx_map[old_i]
            p = parent[old_i]
            active_parent[new_i] = (
                idx_map[p] if p is not None and p in idx_map else None
            )
        return active_nodes, active_parent

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample(self, rng: np.random.Generator) -> np.ndarray:
        """Sample a random point uniformly within :attr:`bounds`.

        Args:
            rng: NumPy random generator to use.

        Returns:
            Random position as a numpy array of shape ``(D,)``.
        """
        lo = np.array([b[0] for b in self.bounds])
        hi = np.array([b[1] for b in self.bounds])
        return rng.uniform(lo, hi)

    def _select_active(
        self,
        nodes: List[np.ndarray],
        active: Set[int],
        x_rand: np.ndarray,
    ) -> Optional[int]:
        """Select the nearest active node to the sample.

        For geometric (no-dynamics) planning, nearest-neighbor selection
        steers the tree toward the random sample most effectively.

        Args:
            nodes: All tree nodes.
            active: Set of active node indices.
            x_rand: Random sample toward which we want to extend.

        Returns:
            Index of the nearest active node, or ``None`` if *active* is
            empty.
        """
        if not active:
            return None
        return min(
            active,
            key=lambda i: float(
                np.linalg.norm((nodes[i] - x_rand) / self.step_size)
            ),
        )

    def _steer(self, from_pt: np.ndarray, to_pt: np.ndarray) -> np.ndarray:
        """Move from *from_pt* toward *to_pt* by at most one normalized step.

        Args:
            from_pt: Origin position.
            to_pt: Target position.

        Returns:
            New position at most one :attr:`step_size` away from *from_pt*
            in normalized space.
        """
        delta = to_pt - from_pt
        dist = float(np.linalg.norm(delta / self.step_size))
        if dist <= 1.0:
            return to_pt.copy()
        return from_pt + delta / dist

    def _nearest_witness(
        self,
        witnesses: List[np.ndarray],
        point: np.ndarray,
    ) -> Optional[int]:
        """Return the index of the nearest witness within *witness_radius*.

        Args:
            witnesses: Current witness positions.
            point: Query position.

        Returns:
            Index of the nearest witness within :attr:`witness_radius`, or
            ``None`` if no such witness exists.
        """
        best_idx: Optional[int] = None
        best_dist = math.inf
        for i, w in enumerate(witnesses):
            d = float(np.linalg.norm((w - point) / self.step_size))
            if d < best_dist and d <= self.witness_radius:
                best_dist = d
                best_idx = i
        return best_idx

    def _segment_free(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Check whether the segment from *a* to *b* is collision-free.

        Args:
            a: Segment start.
            b: Segment end.

        Returns:
            True if no intermediate point is occupied.
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
        """Trace back from *goal_node* to the root.

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
