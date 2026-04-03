"""Unit tests for 3D grid planning scenarios."""

from __future__ import annotations

import numpy as np
import pytest

from arco.mapping import EuclideanGrid
from arco.planning.discrete.astar import AStarPlanner


def test_3d_grid_basic_path():
    """Test A* finds a path in a 3D grid without obstacles."""
    grid = EuclideanGrid(shape=(5, 5, 5))
    planner = AStarPlanner(grid)
    path = planner.plan((0, 0, 0), (4, 4, 4))
    assert path is not None
    assert len(path) > 0
    assert path[0] == (0, 0, 0)
    assert path[-1] == (4, 4, 4)


def test_3d_grid_with_obstacle():
    """Test A* navigates around a 3D obstacle."""
    grid = EuclideanGrid(shape=(5, 5, 5))
    # Place obstacle in center
    grid.set_occupied((2, 2, 2))
    planner = AStarPlanner(grid)
    path = planner.plan((0, 0, 0), (4, 4, 4))
    assert path is not None
    assert (2, 2, 2) not in path


def test_3d_grid_blocked_path():
    """Test A* returns None when goal is completely blocked."""
    grid = EuclideanGrid(shape=(5, 5, 5))
    # Completely surround (4,4,4) in all 26 neighboring cells
    for i in range(3, 5):
        for j in range(3, 5):
            for k in range(3, 5):
                if (i, j, k) != (4, 4, 4):
                    grid.set_occupied((i, j, k))
    planner = AStarPlanner(grid)
    path = planner.plan((0, 0, 0), (4, 4, 4))
    assert path is None


def test_3d_grid_vertical_obstacle():
    """Test A* can navigate around a vertical obstacle column."""
    grid = EuclideanGrid(shape=(10, 10, 10))
    # Create vertical column obstacle
    for z in range(10):
        grid.set_occupied((5, 5, z))
    planner = AStarPlanner(grid)
    path = planner.plan((0, 5, 0), (9, 5, 0))
    assert path is not None
    # Path must avoid the column at (5, 5, z)
    for cell in path:
        assert cell != (5, 5, 0) or cell == (0, 5, 0)  # Can start there


def test_3d_grid_diagonal_movement():
    """Test that 3D EuclideanGrid allows diagonal movement."""
    grid = EuclideanGrid(shape=(5, 5, 5))
    planner = AStarPlanner(grid)
    path = planner.plan((0, 0, 0), (4, 4, 4))
    # Diagonal path should be shorter than axis-aligned path
    # Pure diagonal in 3D from (0,0,0) to (4,4,4) = 5 steps
    assert len(path) <= 6  # Allow one extra step for A*


def test_3d_grid_position_method():
    """Test that Grid.position() works correctly for 3D grids."""
    grid = EuclideanGrid(shape=(5, 5, 5), cell_size=2.0)
    pos = grid.position((1, 2, 3))
    expected = np.array([2.0, 4.0, 6.0])
    assert np.allclose(pos, expected)


def test_3d_grid_heuristic():
    """Test that heuristic uses Euclidean distance in 3D."""
    grid = EuclideanGrid(shape=(5, 5, 5), cell_size=1.0)
    # Euclidean distance from (0,0,0) to (3,4,0) = 5.0
    h = grid.heuristic((0, 0, 0), (3, 4, 0))
    assert np.isclose(h, 5.0)


def test_3d_grid_ground_plane_obstacle():
    """Test planning with obstacles on ground plane."""
    grid = EuclideanGrid(shape=(10, 10, 10))
    # Place a wall on ground plane
    for x in range(3, 7):
        for z in range(5):
            grid.set_occupied((x, 5, z))

    planner = AStarPlanner(grid)
    # Plan from one side to the other
    path = planner.plan((0, 0, 0), (9, 9, 0))
    assert path is not None
    # Path must go around the wall
    for cell in path:
        if cell[0] in range(3, 7) and cell[2] < 5:
            assert cell[1] != 5  # Cannot cross through wall


def test_3d_grid_neighbors_count():
    """Test that 3D EuclideanGrid has correct number of neighbors."""
    grid = EuclideanGrid(shape=(5, 5, 5))
    # Center cell (2,2,2) should have 26 neighbors (3^3 - 1)
    neighbors = list(grid.neighbors((2, 2, 2)))
    assert len(neighbors) == 26


def test_3d_grid_corner_neighbors():
    """Test that corner cells have fewer neighbors."""
    grid = EuclideanGrid(shape=(5, 5, 5))
    # Corner cell (0,0,0) should have 7 neighbors (2^3 - 1)
    neighbors = list(grid.neighbors((0, 0, 0)))
    assert len(neighbors) == 7


def test_3d_planning_with_multiple_obstacles():
    """Test A* with multiple scattered obstacles."""
    grid = EuclideanGrid(shape=(10, 10, 10))
    # Place several obstacles
    obstacles = [
        (2, 2, 2),
        (3, 3, 3),
        (5, 5, 0),
        (7, 2, 1),
        (2, 7, 2),
    ]
    for obs in obstacles:
        grid.set_occupied(obs)

    planner = AStarPlanner(grid)
    path = planner.plan((0, 0, 0), (9, 9, 9))
    assert path is not None
    # Verify path avoids all obstacles
    for obs in obstacles:
        assert obs not in path
