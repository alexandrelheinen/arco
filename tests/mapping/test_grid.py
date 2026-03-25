import numpy as np

from arco.mapping import EuclideanGrid, Grid, ManhattanGrid


def test_grid_basic():
    grid = Grid((3, 3))
    assert grid.shape == (3, 3)
    assert grid.data.shape == (3, 3)
    grid.set_occupied((1, 1))
    assert grid.is_occupied((1, 1))
    grid.set_free((1, 1))
    assert not grid.is_occupied((1, 1))


def test_manhattan_neighbors():
    grid = ManhattanGrid((3, 3))
    neighbors = list(grid.neighbors((1, 1)))
    assert set(neighbors) == {(0, 1), (2, 1), (1, 0), (1, 2)}


def test_euclidean_neighbors():
    grid = EuclideanGrid((3, 3))
    neighbors = set(grid.neighbors((1, 1)))
    expected = {(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)}
    assert neighbors == expected


def test_manhattan_distance():
    grid = ManhattanGrid((0,))
    assert grid.distance((0, 0), (2, 3)) == 5


def test_euclidean_distance():
    grid = EuclideanGrid((0,))
    assert np.isclose(grid.distance((0, 0), (3, 4)), 5.0)
