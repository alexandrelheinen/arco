"""
Base N-dimensional grid for discrete planners (A*, D*, etc).
Implements the Graph interface so that planners can treat grids as graphs.
"""

from __future__ import annotations

import logging
import math
from abc import abstractmethod
from typing import Iterator, Sequence, Tuple

import numpy as np

from arco.mapping.graph import Graph

_log = logging.getLogger(__name__)


class Grid(Graph):
    """N-dimensional grid for discrete planners (A*, D*, etc).

    Inherits from Graph, so it can be used as a graph by planners.
    Each cell is either free (0) or occupied (1).
    Nodes are grid indices (tuples), edges are valid moves (neighbors).

    The grid can be constructed in two ways:

    1. **Cell-based** (legacy): pass *shape* as a sequence of integers.
       ``cell_size`` defaults to 1.0 m.
    2. **Metric**: pass *physical_size* (physical dimensions in meters) and
       *cell_size* (meters per cell).  The number of cells along each
       axis is ``ceil(physical_size[i] / cell_size)``, which is then rounded
       up to the nearest integer satisfying any subclass constraints.
       If the requested *physical_size* is not an exact multiple of
       *cell_size*, the actual physical extent is extended to the next
       multiple of *cell_size* and logged as an approximation.

    Attributes:
        shape: Grid dimensions in cells (rows, cols, …).
        data: Occupancy array (0 = free, 1 = occupied).
        cell_size: Physical size of one cell (meters).
        physical_size: Actual physical extent of the grid in meters per axis.
    """

    shape: Tuple[int, ...]
    data: np.ndarray
    cell_size: float
    physical_size: Tuple[float, ...]

    def __init__(
        self,
        shape: Sequence[int] | None = None,
        *,
        physical_size: Sequence[float] | None = None,
        cell_size: float = 1.0,
    ) -> None:
        """Initialize a grid either by cell shape or by physical dimensions.

        Exactly one of *shape* or *physical_size* must be provided.

        Args:
            shape: Grid dimensions in cells.  Mutually exclusive with
                *physical_size*.
            physical_size: Physical size of the grid in meters for each axis.
                Mutually exclusive with *shape*.  Requires *cell_size*.
            cell_size: Physical size of one cell in meters (default 1.0).
                Used only when *physical_size* is given; ignored otherwise.

        Raises:
            ValueError: If neither or both of *shape* and *physical_size* are
                given, or if *cell_size* is not positive.
        """
        super().__init__()

        if shape is None and physical_size is None:
            raise ValueError(
                "Provide either 'shape' (cells) or 'physical_size' (meters)."
            )
        if shape is not None and physical_size is not None:
            raise ValueError(
                "Provide either 'shape' or 'physical_size', not both."
            )
        if cell_size <= 0:
            raise ValueError(f"cell_size must be positive, got {cell_size!r}.")

        if physical_size is not None:
            # Metric construction: derive cell count from physical size.
            computed: list[int] = []
            actual: list[float] = []
            for i, dim in enumerate(physical_size):
                n_cells = math.ceil(dim / cell_size)
                actual_dim = n_cells * cell_size
                if not math.isclose(actual_dim, dim, rel_tol=1e-6):
                    _log.warning(
                        "Grid axis %d: requested %.6g m is not a multiple "
                        "of cell_size=%.6g m; extended to %.6g m (%d cells).",
                        i,
                        dim,
                        cell_size,
                        actual_dim,
                        n_cells,
                    )
                computed.append(n_cells)
                actual.append(actual_dim)
            self.shape = tuple(computed)
            self.cell_size = float(cell_size)
            self.physical_size = tuple(actual)
        else:
            # Cell-based construction (legacy path).
            self.shape = tuple(shape)  # type: ignore[arg-type]
            self.cell_size = float(cell_size)
            self.physical_size = tuple(s * cell_size for s in self.shape)

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

    def position(self, idx: Tuple[int, ...]) -> np.ndarray:
        """Return the Cartesian position of a grid cell.

        Converts a cell index to a continuous position by multiplying each
        index component by :attr:`cell_size`.

        Args:
            idx: Cell index tuple, e.g. ``(row, col)`` for a 2-D grid.

        Returns:
            Position as a :class:`numpy.ndarray` of shape ``(N,)``.
        """
        return np.array(idx, dtype=float) * self.cell_size

    def heuristic(self, a: Tuple[int, ...], b: Tuple[int, ...]) -> float:
        """Admissible A* heuristic: Euclidean distance between two cells.

        Uses the physical position of each cell (index multiplied by
        :attr:`cell_size`) so the heuristic is expressed in meters and
        correctly accounts for non-unit cell sizes.

        Euclidean distance is always <= the true path cost for standard
        Manhattan (unit step cost 1) and Euclidean (unit step cost sqrt(2))
        grids, so it is admissible for those subclasses.  Subclasses with
        non-unit or non-uniform step costs should override this method with
        a tighter admissible bound.

        Using Euclidean distance instead of the grid's own ``distance``
        method (e.g. Manhattan distance on ``ManhattanGrid``) breaks
        f-score ties that otherwise cause A* to produce L-shaped paths on
        symmetric grids: diagonal cells have strictly smaller Euclidean h
        than off-diagonal cells with the same g-score.

        Args:
            a: First cell index.
            b: Second cell index.

        Returns:
            Euclidean distance as float.
        """
        return float(np.linalg.norm(self.position(a) - self.position(b)))
