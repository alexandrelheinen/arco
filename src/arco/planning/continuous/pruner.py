"""TrajectoryPruner: optimal minimum-waypoint node reduction for raw paths."""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Callable, List, Optional

import numpy as np

from arco.mapping.occupancy import Occupancy

logger = logging.getLogger(__name__)


class TrajectoryPruner:
    """Reduce the node count of a raw path before trajectory optimization.

    The pruner finds the *minimum-waypoint* subsequence of the original path
    such that every consecutive pair of selected nodes can be connected by a
    direct, feasible segment.  Unlike a greedy scan, this BFS-based algorithm
    is provably optimal: the returned path uses the fewest possible nodes.

    The algorithm models path-node indices as vertices in a directed graph.
    There is an edge from index ``i`` to index ``j > i`` whenever the direct
    segment ``path[i] → path[j]`` passes the feasibility check.  A BFS from
    index ``0`` to index ``n-1`` finds the shortest (fewest-hops) path through
    this graph, which maps directly to the minimum-waypoint pruned path.

    The invariant that consecutive nodes in the original path are always
    directly connectable (guaranteed by the planner) ensures the BFS will
    always find a path.

    Args:
        occupancy: Occupancy map used for the default collision check.
        step_size: Per-dimension planner step size as an N-dimensional
            array ``[L_0, L_1, ..., L_{N-1}]``.  Used to compute the
            minimum number of collision-check samples along each pruned
            segment as ``ceil(max_i(D[i] / L[i]))``, where ``D[i]`` is
            the total variation of dimension *i* along the segment.
            Must be strictly positive in every component.
        collision_check_count: Minimum number of intermediate sample
            points per segment used by the built-in collision check.
            The actual count is the maximum of this value and the count
            derived from *step_size*.  Defaults to ``10``.
    """

    def __init__(
        self,
        occupancy: Occupancy,
        step_size: np.ndarray,
        collision_check_count: int = 10,
    ) -> None:
        """Initialize the TrajectoryPruner.

        Args:
            occupancy: Occupancy map for the default segment collision
                check.
            step_size: Per-dimension planner step size vector
                ``[L_0, ..., L_{N-1}]``.  Must be a 1-D numpy array
                (or array-like) with strictly positive components.
                No scalar fallback is provided; every dimension must be
                given explicitly.
            collision_check_count: Minimum number of intermediate sample
                points per segment used by the built-in collision check.

        Raises:
            ValueError: If *collision_check_count* is less than one.
            ValueError: If *step_size* is not a non-empty 1-D array with
                strictly positive elements.
        """
        if collision_check_count < 1:
            raise ValueError(
                "collision_check_count must be at least 1; "
                f"got {collision_check_count}."
            )
        step_size_arr = np.asarray(step_size, dtype=float)
        if step_size_arr.ndim != 1 or step_size_arr.size == 0:
            raise ValueError(
                "step_size must be a non-empty 1-D array; "
                f"got shape {step_size_arr.shape}."
            )
        if np.any(step_size_arr <= 0):
            raise ValueError(
                "step_size elements must be strictly positive; "
                f"got {step_size_arr}."
            )
        self.occupancy = occupancy
        self.step_size: np.ndarray = step_size_arr
        self.collision_check_count = collision_check_count

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prune(
        self,
        path: List[np.ndarray],
        steer: Optional[Callable[[np.ndarray, np.ndarray], bool]] = None,
    ) -> List[np.ndarray]:
        """Return a pruned copy of *path* with the minimum number of nodes.

        Uses a BFS over path indices to find the shortest (fewest-hops)
        subsequence in which every consecutive pair of nodes is directly
        connected by a feasible segment.  This guarantees an optimal
        (minimum-waypoint) result, unlike a greedy scan.

        Args:
            path: Ordered list of position arrays ``[start, ..., goal]``.
                Each element must be a numpy array of the same shape.
                Empty paths and single-node paths are returned unchanged.
            steer: Optional callable ``(a, b) -> bool`` that returns
                ``True`` when a direct connection from *a* to *b* is
                feasible (collision-free and dynamically valid).  When
                ``None``, the built-in linear-interpolation collision
                check is used.

        Returns:
            A new list containing the optimal pruned subset of *path*
            nodes.  The first and last nodes are always preserved.  The
            returned path has a node count less than or equal to the
            input.
        """
        node_count = len(path)
        if node_count <= 2:  # nothing to prune
            return list(path)

        feasible = steer if steer is not None else self._segment_free

        # BFS over indices 0..node_count-1.
        # prev[i] = the index from which BFS first reached i.
        prev: list[int | None] = [None] * node_count
        prev[0] = 0  # sentinel: root is its own predecessor
        queue: deque[int] = deque([0])
        goal = node_count - 1

        while queue:
            current = queue.popleft()
            if current == goal:
                break
            # Explore all forward neighbours — feasibility is not monotone
            # in general (an obstacle may block a short jump but allow a
            # longer one that goes around it), so every candidate must be
            # checked to guarantee an optimal result.
            for nxt in range(current + 1, node_count):
                if prev[nxt] is not None:
                    continue  # already visited
                if feasible(path[current], path[nxt]):
                    prev[nxt] = current
                    queue.append(nxt)
                    if nxt == goal:
                        break

        # Reconstruct the path by tracing predecessors from goal to root.
        indices: list[int] = []
        idx = goal
        while idx != 0:
            indices.append(idx)
            parent = prev[idx]
            if parent is None:
                # Fallback: BFS did not reach the goal (should not happen
                # given the planner invariant).  Return the original path.
                logger.warning(
                    "TrajectoryPruner: BFS did not reach goal; "
                    "returning original path."
                )
                return list(path)
            idx = parent
        indices.append(0)
        indices.reverse()

        pruned = [path[i] for i in indices]

        logger.info(
            "TrajectoryPruner: %d -> %d nodes (%.0f %% reduction)",
            node_count,
            len(pruned),
            100.0 * (1.0 - len(pruned) / node_count),
        )
        return pruned

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _segment_free(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Check whether the segment from *a* to *b* is collision-free.

        Samples the segment at an adaptive density driven by two
        complementary criteria, taking the stricter of the two:

        **Step-size criterion** (primary):
            ``n = ceil(max_i(D[i] / L[i]))`` where ``D[i] = |b[i] - a[i]|``
            is the per-dimension total variation and ``L = self.step_size``.
            This places at least one sample per planner step in every
            dimension, so a pruned shortcut is checked at the same
            resolution the planner used when it built the original path.
            When ``D[i] = 0`` for a dimension, that dimension contributes
            zero to the maximum and is safely ignored.

        **Clearance criterion** (safety floor):
            ``n_clear = ceil(2 * ‖b − a‖ / clearance) + 2`` ensures spacing
            ≤ ``clearance / 2`` so that any obstacle sphere (radius
            *clearance* in the raw occupancy space) along the segment is hit
            by at least one sample.

        The final sample count is ``max(n_step + 1, n_clear,
        collision_check_count + 2, 2)``.

        For each sample the distance to the nearest obstacle is queried
        and compared against the clearance threshold — the
        *nearest + distance-thresholding* pattern mandated by the
        :class:`~arco.mapping.KDTreeOccupancy` contract.  When
        :meth:`~arco.mapping.KDTreeOccupancy.query_distances` is
        available, all distances are fetched in a single batch call for
        efficiency.

        Args:
            a: Segment start position (raw C-space units).
            b: Segment end position (raw C-space units).

        Returns:
            ``True`` if every sampled point is at least *clearance* away
            from all obstacle points; ``False`` if any point violates the
            clearance constraint.
        """
        clearance: float = getattr(self.occupancy, "clearance", 0.0)

        # --- Step-size criterion (primary) --------------------------------
        # D[i] / L[i] gives the number of planner steps spanned by this
        # segment in dimension i.  When D[i] = 0 the ratio is 0 and does
        # not contribute to the maximum.
        D = np.abs(b - a)
        ndim = min(D.size, self.step_size.size)
        ratios = D[:ndim] / self.step_size[:ndim]
        n_step = (
            int(math.ceil(float(np.max(ratios)))) if np.any(ratios > 0) else 0
        )

        # --- Clearance criterion (safety floor) ---------------------------
        length = float(np.linalg.norm(b - a))
        if clearance > 0.0 and length > 0.0:
            n_clear = int(math.ceil(2.0 * length / clearance)) + 2
        else:
            n_clear = 0

        n_samples = max(n_step + 1, n_clear, self.collision_check_count + 2, 2)

        ts = np.linspace(0.0, 1.0, n_samples)
        pts = a + ts[:, np.newaxis] * (b - a)  # shape (n_samples, D)

        # Prefer the batch distance query for efficiency; the explicit
        # comparison `distances >= clearance` is the canonical
        # nearest + distance-thresholding check.
        if clearance > 0.0 and hasattr(self.occupancy, "query_distances"):
            distances = self.occupancy.query_distances(pts)
            return bool(np.all(distances >= clearance))

        # Fallback for occupancy maps that only expose is_occupied().
        # is_occupied() also performs nearest + distance thresholding.
        return all(not self.occupancy.is_occupied(pt) for pt in pts)
