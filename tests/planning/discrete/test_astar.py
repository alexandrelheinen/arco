import numpy as np
import pytest

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


def test_astar_planner_occupied_start_returns_none():
    """A* must return None when the start cell is occupied."""
    grid = ManhattanGrid((5, 5))
    grid.set_occupied((0, 0))  # start is blocked
    planner = AStarPlanner(grid)
    path = planner.plan((0, 0), (4, 4))
    assert path is None


def test_astar_planner_occupied_goal_returns_none():
    """A* must return None when the goal cell is occupied."""
    grid = ManhattanGrid((5, 5))
    grid.set_occupied((4, 4))  # goal is blocked
    planner = AStarPlanner(grid)
    path = planner.plan((0, 0), (4, 4))
    assert path is None


def test_astar_planner_start_equals_goal():
    """When start == goal A* should return a single-element path."""
    grid = ManhattanGrid((5, 5))
    planner = AStarPlanner(grid)
    path = planner.plan((2, 2), (2, 2))
    assert path is not None
    assert path[0] == (2, 2)
    assert path[-1] == (2, 2)


def test_astar_path_simplification_straight_corridor():
    """A* path simplification must collapse straight runs to endpoints."""
    grid = ManhattanGrid((1, 6))
    planner = AStarPlanner(grid)
    start = (0, 0)
    goal = (0, 5)

    path = planner.plan(start, goal)

    assert path is not None
    assert path == [start, goal]


def test_astar_plan_with_diagnostics_returns_tree_data():
    """Diagnostics mode must expose expansion order and parent links."""
    grid = ManhattanGrid((4, 4))
    planner = AStarPlanner(grid)
    start = (0, 0)
    goal = (3, 3)

    path, expanded_order, came_from = planner.plan_with_diagnostics(
        start, goal
    )

    assert path is not None
    assert path[0] == start
    assert path[-1] == goal
    assert len(expanded_order) > 0
    assert start in expanded_order
    assert goal in expanded_order
    assert isinstance(came_from, dict)


def test_astar_plan_with_diagnostics_occupied_start():
    """Diagnostics mode must return empty collections when start is occupied."""
    grid = ManhattanGrid((5, 5))
    grid.set_occupied((0, 0))
    planner = AStarPlanner(grid)
    path, expanded, came_from = planner.plan_with_diagnostics((0, 0), (4, 4))
    assert path is None
    assert expanded == []
    assert came_from == {}
