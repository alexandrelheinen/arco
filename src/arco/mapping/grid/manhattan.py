"""ManhattanGrid: axis-aligned neighbor logic for discrete planners."""

from __future__ import annotations

from typing import Iterator, Sequence, Tuple

import numpy as np

from .base import Grid


class ManhattanGrid(Grid):
    """
    Grid with Manhattan (L1) connectivity and distance.

    Only axis-aligned neighbors are considered.
    """

    def __init__(self, shape: Sequence[int]) -> None:
        """Initialize a ManhattanGrid.

        Args:
            shape: The shape of the grid as a sequence of integers.
        """
        super().__init__(shape)

    def distance(self, a: Tuple[int, ...], b: Tuple[int, ...]) -> int:
        """Return L1 (Manhattan) distance between two nodes.

        Args:
            a: First node index.
            b: Second node index.
        Returns:
            Manhattan (L1) distance as int.
        """
        return sum(abs(x - y) for x, y in zip(a, b))

    def neighbors(self, idx: Tuple[int, ...]) -> Iterator[Tuple[int, ...]]:
        """Yield axis-aligned neighbor indices for a given cell.

        Args:
            idx: Index of the cell to find neighbors for.
        Yields:
            Neighbor indices as tuples.
        """
        ndim = len(self.shape)
        idx_arr = np.array(idx)
        for d in range(ndim):
            for sign in [-1, 1]:
                offset = np.zeros(ndim, dtype=int)
                offset[d] = sign
                neighbor = tuple(idx_arr + offset)
                if all(0 <= n < s for n, s in zip(neighbor, self.shape)):
                    yield neighbor

        def squared_distance(
            self, a: Tuple[int, ...], b: Tuple[int, ...]
        ) -> int:
            """Return squared L1 (Manhattan) distance between two nodes.

            Args:
                a: First node index.
                b: Second node index.
            Returns:
                Squared L1 distance as integer.
            """
            d = sum(abs(x - y) for x, y in zip(a, b))
            return d * d
