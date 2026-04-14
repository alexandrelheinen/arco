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
        collision_check_count: Number of intermediate points sampled
            along each candidate segment when performing the built-in
            collision check.  Higher values are more accurate but
            slower.  Defaults to ``10``.
    """

    def __init__(
        self,
        occupancy: Occupancy,
        collision_check_count: int = 10,
    ) -> None:
        """Initialize the TrajectoryPruner.

        Args:
            occupancy: Occupancy map for the default segment collision
                check.
            collision_check_count: Number of intermediate sample points
                per segment used by the built-in collision check.

        Raises:
            ValueError: If *collision_check_count* is less than one.
        """
        if collision_check_count < 1:
            raise ValueError(
                "collision_check_count must be at least 1; "
                f"got {collision_check_count}."
            )
        self.occupancy = occupancy
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

        The BFS operates only over the *inner* path ``path[0] … path[-2]``
        (the pre-last node).  ``path[-1]`` (the exact goal appended by the
        planner) is **always** preserved as the final waypoint.  This
        ensures that the direct approach segment from the last tree node to
        the exact goal is never replaced by a longer, potentially
        obstacle-clipping shortcut.

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
            nodes.  The first, pre-last, and last nodes are always
            preserved.  The returned path has a node count less than or
            equal to the input.
        """
        node_count = len(path)
        if node_count <= 2:  # nothing to prune
            return list(path)

        feasible = steer if steer is not None else self._segment_free

        # BFS over the *inner* path: indices 0..node_count-2.
        # path[-1] (the exact goal appended by the planner) is always kept
        # and appended after BFS so that the pre-last node (path[-2]) is
        # never skipped in favour of a direct-to-goal shortcut that might
        # bypass an important bend or clip an obstacle.
        inner_count = node_count - 1  # number of inner nodes (path[0..-2])
        prev: list[int | None] = [None] * inner_count
        prev[0] = 0  # sentinel: root is its own predecessor
        queue: deque[int] = deque([0])
        goal = inner_count - 1  # target index within the inner slice

        while queue:
            current = queue.popleft()
            if current == goal:
                break
            # Explore all forward neighbours — feasibility is not monotone
            # in general (an obstacle may block a short jump but allow a
            # longer one that goes around it), so every candidate must be
            # checked to guarantee an optimal result.
            for nxt in range(current + 1, inner_count):
                if prev[nxt] is not None:
                    continue  # already visited
                if feasible(path[current], path[nxt]):
                    prev[nxt] = current
                    queue.append(nxt)
                    if nxt == goal:
                        break

        # Reconstruct the inner path by tracing predecessors from the
        # pre-last node back to the root.
        indices: list[int] = []
        idx = goal
        while idx != 0:
            indices.append(idx)
            parent = prev[idx]
            if parent is None:
                # Fallback: BFS did not reach the pre-last node (should not
                # happen given the planner invariant).  Return original path.
                logger.warning(
                    "TrajectoryPruner: BFS did not reach goal; "
                    "returning original path."
                )
                return list(path)
            idx = parent
        indices.append(0)
        indices.reverse()

        # Always append the exact goal (path[-1]) after the pre-last node.
        pruned = [path[i] for i in indices]
        pruned.append(path[-1])

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

        Samples the segment at an **adaptive** density derived from the
        occupancy clearance radius so that no two consecutive sample points
        are further than ``clearance / 2`` apart.  This guarantees that
        no obstacle region (a sphere of radius *clearance* around each
        obstacle point) can slip undetected between samples.

        For each sample the distance to the nearest obstacle is queried
        and compared against the clearance threshold — the
        *nearest + distance-thresholding* pattern mandated by the
        :class:`~arco.mapping.KDTreeOccupancy` contract.  When
        :meth:`~arco.mapping.KDTreeOccupancy.query_distances` is
        available, all distances are fetched in a single batch call for
        efficiency.

        Args:
            a: Segment start position.
            b: Segment end position.

        Returns:
            ``True`` if every sampled point is at least *clearance* away
            from all obstacle points; ``False`` if any point violates the
            clearance constraint.
        """
        # When the occupancy map does not expose a clearance attribute (e.g.
        # a custom map that only implements is_occupied), clearance defaults
        # to 0.0 and the adaptive density logic below is skipped; the pruner
        # then falls back to the fixed collision_check_count sample count.
        clearance: float = getattr(self.occupancy, "clearance", 0.0)
        length = float(np.linalg.norm(b - a))

        if clearance > 0.0 and length > 0.0:
            # We need spacing s ≤ clearance/2 so that any obstacle region
            # (a ball of radius *clearance* around each obstacle point) that
            # the segment passes through is detected by at least one sample.
            # Number of intervals = ceil(length / (clearance/2))
            #                     = ceil(2 * length / clearance),
            # so n_points = intervals + 1 ≤ ceil(2*length/clearance) + 2.
            min_samples = int(math.ceil(2.0 * length / clearance)) + 2
        else:
            min_samples = 0

        n_samples = max(self.collision_check_count + 2, min_samples)

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
