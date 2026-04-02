"""EuclideanGrid: diagonal neighbor logic for discrete planners."""

from __future__ import annotations

from typing import Iterator, Sequence, Tuple

import numpy as np

from .base import Grid


class EuclideanGrid(Grid):
    """
    Grid with diagonal (L2) connectivity and distance.

    Includes diagonal neighbors and uses Euclidean distance.
    """

    def __init__(self, shape: Sequence[int]) -> None:
        """Initialize a EuclideanGrid.

        Args:
            shape: The shape of the grid as a sequence of integers.
        """
        super().__init__(shape)

    def distance(self, a: Tuple[int, ...], b: Tuple[int, ...]) -> float:
        """Return L2 (Euclidean) distance between two nodes.

        Args:
            a: First node index.
            b: Second node index.
        Returns:
            Euclidean (L2) distance as float.
        """
        return float(np.linalg.norm(np.array(a) - np.array(b)))

    def neighbors(self, idx: Tuple[int, ...]) -> Iterator[Tuple[int, ...]]:
        """Yield diagonal (including axis-aligned) neighbor indices for a given cell.

        Args:
            idx: Index of the cell to find neighbors for.
        Yields:
            Neighbor indices as tuples.
        """
        ndim = len(self.shape)
        idx_arr = np.array(idx)
        for x in np.ndindex(*(3,) * ndim):
            if all(xi == 1 for xi in x):
                continue  # skip the center cell itself
            offset = np.array(x) - 1
            neighbor = tuple(idx_arr + offset)
            if all(0 <= n < s for n, s in zip(neighbor, self.shape)):
                yield neighbor

        def squared_distance(
            self, a: Tuple[int, ...], b: Tuple[int, ...]
        ) -> int:
            """Return squared L2 (Euclidean) distance between two nodes (no sqrt).

            Args:
                a: First node index.
                b: Second node index.
            Returns:
                Squared L2 distance as integer.
            """
            return int(np.sum((np.array(a) - np.array(b)) ** 2))
