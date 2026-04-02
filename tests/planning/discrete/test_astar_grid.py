"""
Tests for A* on grids: path correctness and staircase property.

On a symmetric empty Manhattan grid from (0, 0) to (N-1, N-1) the
Euclidean heuristic used by AStarPlanner guides it toward the diagonal,
so the returned path is a staircase: every intermediate cell satisfies
|row - col| <= 1.
"""

import numpy as np
import pytest

from arco.mapping import ManhattanGrid
from arco.planning.discrete.astar import AStarPlanner


def test_astar_empty_grid_staircase():
    """A* on an empty Manhattan grid should return a staircase path.

    On a symmetric NxN grid from (0,0) to (N-1,N-1) the optimal cost is
    2*(N-1) steps regardless of path shape.  With the Euclidean heuristic
    A* prefers cells on the straight line from start to goal, producing a
    staircase where |row - col| <= 1 at every step.
    """
    n = 10
    grid = ManhattanGrid((n, n))
    start = (0, 0)
    goal = (n - 1, n - 1)
    planner = AStarPlanner(grid)
    path = planner.plan(start, goal)

    assert path is not None, "A* did not find a path on empty grid"
    assert path[0] == start
    assert path[-1] == goal

    # Optimal path length is 2*(n-1)+1 nodes
    assert len(path) == 2 * (n - 1) + 1

    # All moves must be axis-aligned (Manhattan connectivity)
    for a, b in zip(path[:-1], path[1:]):
        dr = abs(a[0] - b[0])
        dc = abs(a[1] - b[1])
        assert (dr == 1 and dc == 0) or (
            dr == 0 and dc == 1
        ), f"Non-Manhattan move: {a} -> {b}"

    # Staircase property: the path never drifts more than 1 cell from the
    # main diagonal — this rules out the L-shaped path that naive A*
    # (lexicographic tie-breaking) would produce.
    max_deviation = max(abs(int(r) - int(c)) for r, c in path)
    assert max_deviation <= 1, (
        f"Path is not a staircase (max |row-col| = {max_deviation}); "
        f"A* is producing an L-shape instead of navigating toward the goal."
    )


def test_astar_diagonal_path():
    """Legacy test: optimal length and axis-aligned moves on empty grid."""
    n = 10
    grid = ManhattanGrid((n, n))
    start = (0, 0)
    goal = (n - 1, n - 1)
    planner = AStarPlanner(grid)
    path = planner.plan(start, goal)

    assert path is not None, "A* did not find a path on empty grid"
    assert len(path) == 2 * (n - 1) + 1
    assert path[0] == start
    assert path[-1] == goal
    for a, b in zip(path[:-1], path[1:]):
        dr = abs(a[0] - b[0])
        dc = abs(a[1] - b[1])
        assert (dr == 1 and dc == 0) or (
            dr == 0 and dc == 1
        ), f"Non-Manhattan move: {a} -> {b}"
