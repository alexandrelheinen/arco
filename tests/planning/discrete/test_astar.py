import numpy as np

from arco.mapping import ManhattanGrid
from arco.planning.discrete import AStarPlanner


def test_astar_planner_simple():
    grid = ManhattanGrid((5, 5))
    grid.set_occupied((2, 1))
    grid.set_occupied((2, 2))
    grid.set_occupied((2, 3))
    planner = AStarPlanner(grid)
    start = (0, 0)
    goal = (4, 4)
    path = planner.plan(start, goal)
    assert path is not None
    assert path[0] == start
    assert path[-1] == goal
    for node in path:
        assert not grid.is_occupied(node)


def test_astar_planner_no_path():
    grid = ManhattanGrid((3, 3))
    for i in range(3):
        for j in range(3):
            grid.set_occupied((i, j))
    grid.set_free((0, 0))
    grid.set_free((2, 2))
    planner = AStarPlanner(grid)
    path = planner.plan((0, 0), (2, 2))
    assert path is None
