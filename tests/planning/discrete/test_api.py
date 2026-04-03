import numpy as np
import pytest

from arco.planning import AStar, DStarLite


def test_planning_api_imports():
    grid = np.zeros((3, 3), dtype=int)
    astar = AStar(grid)
    dstar = DStarLite(grid)
    assert hasattr(astar, "search")
    assert hasattr(dstar, "search")


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


@pytest.mark.xfail(
    reason="D* planner not yet implemented",
    strict=True,
    raises=NotImplementedError,
)
def test_dstar_simple():
    grid = np.zeros((5, 5), dtype=int)
    grid[2, 1:4] = 1  # Add a wall
    dstar = DStarLite(grid)
    start = (0, 0)
    goal = (4, 4)
    path = dstar.search(start, goal)
    assert path is not None
    assert path[0] == start
    assert path[-1] == goal
    for node in path:
        assert grid[node] == 0


@pytest.mark.xfail(
    reason="D* planner not yet implemented",
    strict=True,
    raises=NotImplementedError,
)
def test_dstar_no_path():
    grid = np.ones((3, 3), dtype=int)
    grid[0, 0] = 0
    grid[2, 2] = 0
    dstar = DStarLite(grid)
    path = dstar.search((0, 0), (2, 2))
    assert path is None
