"""TrajectoryPruner: greedy forward-scan node reduction for raw paths."""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

import numpy as np

from arco.mapping.occupancy import Occupancy

logger = logging.getLogger(__name__)


class TrajectoryPruner:
    """Reduce the node count of a raw path before trajectory optimization.

    The pruner applies a greedy forward-scan to skip intermediate nodes that
    are topologically redundant — i.e., nodes that can be bypassed by a
    direct, feasible connection.  The algorithm relies on the invariant that
    consecutive nodes in the original path are always directly connectable
    (guaranteed by the planner), so the pruned path is guaranteed to be
    valid.

    Algorithm (greedy forward scan):

    1. Start at anchor ``k = 0``.
    2. Attempt a direct connection from node ``k`` to node ``k + 2``
       (skipping ``k + 1``) using the *steer* callable.
    3. If feasible, advance the lookahead: try ``k → k + 3``,
       ``k → k + 4``, and so on.
    4. When the connection fails (or the end is reached), record node
       ``k + (lookahead - 1)`` in the pruned path, set it as the new
       anchor, and restart from step 2.
    5. The final node is always included.

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
        """Return a pruned copy of *path* with redundant nodes removed.

        Args:
            path: Ordered list of position arrays ``[start, …, goal]``.
                Each element must be a numpy array of the same shape.
                Empty paths and single-node paths are returned unchanged.
            steer: Optional callable ``(a, b) -> bool`` that returns
                ``True`` when a direct connection from *a* to *b* is
                feasible (collision-free and dynamically valid).  When
                ``None``, the built-in linear-interpolation collision
                check is used.

        Returns:
            A new list containing the pruned subset of *path* nodes.
            The first and last nodes are always preserved.  The returned
            path has a node count less than or equal to the input.
        """
        node_count = len(path)
        if node_count <= 2:  # nothing to prune
            return list(path)

        feasible = steer if steer is not None else self._segment_free

        pruned: List[np.ndarray] = [path[0]]
        anchor = 0

        while True:
            # Try advancing the lookahead as far as possible.
            lookahead = 2
            while anchor + lookahead < node_count:
                target_idx = anchor + lookahead
                if feasible(path[anchor], path[target_idx]):
                    lookahead += 1
                else:
                    break

            # The last successfully reached index before the failure.
            best_idx = anchor + lookahead - 1

            if best_idx >= node_count - 1:
                # Reached or passed the last node — we are done.
                break

            pruned.append(path[best_idx])
            anchor = best_idx

        pruned.append(path[-1])

        logger.debug(
            "TrajectoryPruner: %d → %d nodes (%.0f %% reduction)",
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

        Samples ``collision_check_count + 2`` evenly spaced points along
        the segment (including the endpoints) and tests each one against
        the occupancy map.

        Args:
            a: Segment start position.
            b: Segment end position.

        Returns:
            ``True`` if every sampled point is free; ``False`` if any
            point is occupied.
        """
        for t in np.linspace(0.0, 1.0, self.collision_check_count + 2):
            pt = a + t * (b - a)
            if self.occupancy.is_occupied(pt):
                return False
        return True
