

"""
Base N-dimensional grid for discrete planners (A*, D*, etc).
Implements the Graph interface so that planners can treat grids as graphs.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Iterator, Sequence, Tuple


import numpy as np
from arco.mapping.graph import Graph


class Grid(Graph):
    """
    N-dimensional grid for discrete planners (A*, D*, etc).

    Inherits from Graph, so it can be used as a graph by planners.
    Each cell is either free (0) or occupied (1).
    Nodes are grid indices (tuples), edges are valid moves (neighbors).
    """

    shape: Tuple[int, ...]
    data: np.ndarray

    def __init__(self, shape: Sequence[int]) -> None:
        """
        Initialize a grid of given shape.

        Args:
            shape: The shape of the grid as a sequence of integers.
        """
        super().__init__()
        self.shape = tuple(shape)
        self.data = np.zeros(self.shape, dtype=np.uint8)

    def set_occupied(self, idx: Tuple[int, ...]) -> None:
        """
        Mark a cell as occupied.

        Args:
            idx: Index of the cell to mark as occupied.
        """
        self.data[idx] = 1

    def set_free(self, idx: Tuple[int, ...]) -> None:
        """
        Mark a cell as free.

        Args:
            idx: Index of the cell to mark as free.
        """
        self.data[idx] = 0

    def is_occupied(self, idx: Tuple[int, ...]) -> bool:
        """
        Return True if the cell is occupied.

        Args:
            idx: Index of the cell to check.
        Returns:
            True if occupied, False otherwise.
        """
        return self.data[idx] == 1

    @abstractmethod
    def neighbors(self, idx: Tuple[int, ...]) -> Iterator[Tuple[int, ...]]:
        """
        Yield neighbor indices for a given cell (node).

        Args:
            idx: Index of the cell (node) to find neighbors for.
        Yields:
            Neighbor indices as tuples.
        """
        pass

        @abstractmethod
        def squared_distance(self, a: Tuple[int, ...], b: Tuple[int, ...]) -> int:
            """
            Return the squared distance between two nodes (integer, for fast comparison).

            Args:
                a: First node index.
                b: Second node index.
            Returns:
                Squared distance as integer (L1^2 or L2^2 depending on grid type).
            """
            pass
