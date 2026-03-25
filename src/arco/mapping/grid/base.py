"""
Base N-dimensional grid for discrete planners (A*, D*, etc).
"""

import numpy as np


class Grid:
    """
    N-dimensional grid for discrete planners (A*, D*, etc).
    Each cell is either free (0) or occupied (1).
    Supports axis-aligned neighbor queries in all dimensions.
    """

    def __init__(self, shape):
        """Initialize a grid of given shape."""
        self.shape = tuple(shape)
        self.data = np.zeros(self.shape, dtype=np.uint8)

    def set_occupied(self, idx):
        """Mark a cell as occupied."""
        self.data[idx] = 1

    def set_free(self, idx):
        """Mark a cell as free."""
        self.data[idx] = 0

    def is_occupied(self, idx):
        """Return True if the cell is occupied."""
        return self.data[idx] == 1

    def neighbors(self, idx, diagonal=False):
        """
        Yield neighbor indices for a given cell.
        If diagonal=True, includes diagonal neighbors.
        """
        idx = np.array(idx)
        offsets = self._neighbor_offsets(diagonal)
        for offset in offsets:
            neighbor = tuple(idx + offset)
            if all(0 <= n < s for n, s in zip(neighbor, self.shape)):
                yield neighbor

    def _neighbor_offsets(self, diagonal):
        """Return list of neighbor offsets for the grid."""
        ndim = len(self.shape)
        if diagonal:
            offsets = [
                np.array(x)
                for x in np.ndindex(*(3,) * ndim)
                if any(xi != 1 for xi in x)
            ]
            offsets = [o - 1 for o in offsets]
        else:
            offsets = []
            for d in range(ndim):
                for sign in [-1, 1]:
                    offset = np.zeros(ndim, dtype=int)
                    offset[d] = sign
                    offsets.append(offset)
        return offsets
