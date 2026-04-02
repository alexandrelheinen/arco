"""Unit tests for Grid, ManhattanGrid, and EuclideanGrid classes."""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest

from arco.mapping import EuclideanGrid, Grid, ManhattanGrid


def test_grid_basic():
    """Test basic Grid creation and occupancy methods."""
    grid = Grid((3, 3))
    assert grid.shape == (3, 3)
    assert grid.data.shape == (3, 3)
    grid.set_occupied((1, 1))
    assert grid.is_occupied((1, 1))
    grid.set_free((1, 1))
    assert not grid.is_occupied((1, 1))


def test_manhattan_neighbors():
    """Test ManhattanGrid neighbor generation (axis-aligned)."""
    grid = ManhattanGrid((3, 3))
    neighbors = list(grid.neighbors((1, 1)))
    assert set(neighbors) == {(0, 1), (2, 1), (1, 0), (1, 2)}


def test_euclidean_neighbors():
    """Test EuclideanGrid neighbor generation (diagonal included)."""
    grid = EuclideanGrid((3, 3))
    neighbors = set(grid.neighbors((1, 1)))
    expected = {(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)}
    assert neighbors == expected


def test_manhattan_distance():
    """Test ManhattanGrid L1 distance calculation."""
    grid = ManhattanGrid((0,))
    assert grid.distance((0, 0), (2, 3)) == 5


def test_euclidean_distance():
    """Test EuclideanGrid L2 distance calculation."""
    grid = EuclideanGrid((0,))
    assert np.isclose(grid.distance((0, 0), (3, 4)), 5.0)


# ---------------------------------------------------------------------------
# Metric constructor tests
# ---------------------------------------------------------------------------


def test_grid_metric_exact_multiple():
    """Grid from metric dimensions that are exact multiples of cell_size."""
    # 51 m x 51 m, 1 m cells → 51 x 51 cells
    grid = ManhattanGrid(size_m=[51.0, 51.0], cell_size=1.0)
    assert grid.shape == (51, 51)
    assert math.isclose(grid.cell_size, 1.0)
    assert grid.size_m == (51.0, 51.0)


def test_grid_metric_sub_cell():
    """Grid from metric dimensions with sub-metre cell size."""
    # 51 m x 51 m, 0.5 m cells → 102 x 102 cells
    grid = EuclideanGrid(size_m=[51.0, 51.0], cell_size=0.5)
    assert grid.shape == (102, 102)
    assert math.isclose(grid.cell_size, 0.5)
    assert grid.size_m == (51.0, 51.0)


def test_grid_metric_rounds_up(caplog):
    """Grid size is extended upward when not a multiple of cell_size."""
    # 100 m, 3 m cells → ceil(100/3) = 34 cells → 102 m actual
    with caplog.at_level(logging.WARNING, logger="arco.mapping.grid.base"):
        grid = ManhattanGrid(size_m=[100.0, 100.0], cell_size=3.0)

    assert grid.shape == (34, 34)
    assert math.isclose(grid.size_m[0], 102.0)
    assert math.isclose(grid.size_m[1], 102.0)
    # A warning must have been emitted for each extended axis.
    assert len(caplog.records) == 2


def test_grid_cell_based_has_default_cell_size():
    """Cell-based grids report cell_size=1.0 and size_m equal to shape."""
    grid = ManhattanGrid((5, 7))
    assert math.isclose(grid.cell_size, 1.0)
    assert grid.size_m == (5.0, 7.0)


def test_grid_invalid_both_shape_and_size_m():
    """Providing both shape and size_m raises ValueError."""
    with pytest.raises(ValueError, match="not both"):
        Grid((3, 3), size_m=[3.0, 3.0])


def test_grid_invalid_neither():
    """Providing neither shape nor size_m raises ValueError."""
    with pytest.raises(ValueError, match="either"):
        Grid()


def test_grid_invalid_negative_cell_size():
    """Non-positive cell_size raises ValueError."""
    with pytest.raises(ValueError, match="positive"):
        Grid(size_m=[10.0, 10.0], cell_size=-1.0)
