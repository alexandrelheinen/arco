import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)
import numpy as np

from arco.planning import AStar


def test_astar_simple():
    grid = np.zeros((5, 5), dtype=int)
    grid[2, 1:4] = 1  # Add a wall
    astar = AStar(grid)
    start = (0, 0)
    goal = (4, 4)
    path = astar.search(start, goal)
    assert path is not None
    assert path[0] == start
    assert path[-1] == goal
    # Path should not go through the wall
    for node in path:
        assert grid[node] == 0


def test_astar_no_path():
    grid = np.ones((3, 3), dtype=int)
    grid[0, 0] = 0
    grid[2, 2] = 0
    astar = AStar(grid)
    path = astar.search((0, 0), (2, 2))
    assert path is None
