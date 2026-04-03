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
    grid = ManhattanGrid(physical_size=[51.0, 51.0], cell_size=1.0)
    assert grid.shape == (51, 51)
    assert math.isclose(grid.cell_size, 1.0)
    assert grid.physical_size == (51.0, 51.0)


def test_grid_metric_sub_cell():
    """Grid from metric dimensions with sub-metre cell size."""
    # 51 m x 51 m, 0.5 m cells → 102 x 102 cells
    grid = EuclideanGrid(physical_size=[51.0, 51.0], cell_size=0.5)
    assert grid.shape == (102, 102)
    assert math.isclose(grid.cell_size, 0.5)
    assert grid.physical_size == (51.0, 51.0)


def test_grid_metric_rounds_up(caplog):
    """Grid size is extended upward when not a multiple of cell_size."""
    # 100 m, 3 m cells → ceil(100/3) = 34 cells → 102 m actual
    with caplog.at_level(logging.WARNING, logger="arco.mapping.grid.base"):
        grid = ManhattanGrid(physical_size=[100.0, 100.0], cell_size=3.0)

    assert grid.shape == (34, 34)
    assert math.isclose(grid.physical_size[0], 102.0)
    assert math.isclose(grid.physical_size[1], 102.0)
    # A warning must have been emitted for each extended axis.
    assert len(caplog.records) == 2


def test_grid_cell_based_has_default_cell_size():
    """Cell-based grids report cell_size=1.0 and physical_size equal to shape."""
    grid = ManhattanGrid((5, 7))
    assert math.isclose(grid.cell_size, 1.0)
    assert grid.physical_size == (5.0, 7.0)


def test_grid_invalid_both_shape_and_physical_size():
    """Providing both shape and physical_size raises ValueError."""
    with pytest.raises(ValueError, match="not both"):
        Grid((3, 3), physical_size=[3.0, 3.0])


def test_grid_invalid_neither():
    """Providing neither shape nor physical_size raises ValueError."""
    with pytest.raises(ValueError, match="either"):
        Grid()


def test_grid_invalid_negative_cell_size():
    """Non-positive cell_size raises ValueError."""
    with pytest.raises(ValueError, match="positive"):
        Grid(physical_size=[10.0, 10.0], cell_size=-1.0)


# ---------------------------------------------------------------------------
# Grid.position() method
# ---------------------------------------------------------------------------


def test_grid_position_2d():
    """position() converts cell index to Cartesian coordinates."""
    grid = ManhattanGrid((5, 5), cell_size=2.0)
    pos = grid.position((1, 3))
    assert np.allclose(pos, [2.0, 6.0])


def test_grid_position_unit_cell_size():
    """position() with cell_size=1.0 should equal index as floats."""
    grid = EuclideanGrid((4, 4))
    pos = grid.position((2, 3))
    assert np.allclose(pos, [2.0, 3.0])


def test_grid_position_3d():
    """position() works for 3-D grids."""
    grid = ManhattanGrid((3, 3, 3), cell_size=1.0)
    pos = grid.position((1, 2, 0))
    assert np.allclose(pos, [1.0, 2.0, 0.0])


def test_grid_heuristic_respects_cell_size():
    """heuristic() must account for cell_size in the distance calculation."""
    # 2-D grid, cell_size=2.0 → positions scaled by 2
    grid = ManhattanGrid((5, 5), cell_size=2.0)
    # Cells (0,0) → (0,0), (2,3) → (4,6)
    # Euclidean distance = sqrt(16 + 36) = sqrt(52)
    expected = math.sqrt(52)
    assert math.isclose(grid.heuristic((0, 0), (2, 3)), expected)


def test_grid_heuristic_unit_cell_equals_euclidean():
    """With cell_size=1.0, heuristic equals plain Euclidean index distance."""
    grid = EuclideanGrid((5, 5))
    assert math.isclose(grid.heuristic((0, 0), (3, 4)), 5.0)
