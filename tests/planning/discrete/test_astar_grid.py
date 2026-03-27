"""
Test for A* on a Manhattan grid: path should be a straight diagonal (alternating x/y steps).
"""
import numpy as np
import pytest
from arco.mapping import ManhattanGrid
from arco.planning.discrete.astar import AStarPlanner

def test_astar_diagonal_path():
    length = 10.0
    resolution = 1.0
    n = int(length / resolution)
    grid = ManhattanGrid((n, n))
    start = (0, 0)
    goal = (n - 1, n - 1)
    planner = AStarPlanner(grid)
    path = planner.plan(start, goal)
    assert path is not None, "A* did not find a path on empty grid"
    # Path should alternate x/y steps, length should be 2*(n-1)+1
    assert len(path) == 2 * (n - 1) + 1
    # Path should start and end at correct points
    assert path[0] == start
    assert path[-1] == goal
    # All steps should be axis-aligned (no diagonal moves)
    for (a, b) in zip(path[:-1], path[1:]):
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        assert (dx == 1 and dy == 0) or (dx == 0 and dy == 1), f"Non-Manhattan move: {a}->{b}"
