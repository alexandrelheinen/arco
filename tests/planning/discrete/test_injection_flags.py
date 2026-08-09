"""Tests for A* flags and RouteRouter planner injection."""

from __future__ import annotations

import numpy as np

from arco.mapping import ManhattanGrid
from arco.mapping.graph.cartesian import CartesianGraph
from arco.planning.discrete import AStarPlanner, RouteRouter


def test_astar_simplify_path_can_be_disabled():
    grid = ManhattanGrid((1, 6))
    planner = AStarPlanner(grid, simplify_path=False)
    path = planner.plan((0, 0), (0, 5))
    assert path is not None
    assert len(path) == 6


def test_astar_prefer_straight_false_still_finds_path():
    grid = ManhattanGrid((5, 5))
    planner = AStarPlanner(grid, prefer_straight=False)
    path = planner.plan((0, 0), (4, 4))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (4, 4)


def test_route_router_accepts_custom_planner():
    g = CartesianGraph()
    g.add_node(0, 0.0, 0.0)
    g.add_node(1, 1.0, 0.0)
    g.add_edge(0, 1)

    calls = {"n": 0}

    class _Spy:
        def plan(self, start, goal):
            calls["n"] += 1
            return [start, goal]

    router = RouteRouter(g, planner=_Spy())
    result = router.plan(np.array([0.0, 0.0]), np.array([1.0, 0.0]))
    assert result is not None
    assert calls["n"] == 1
    assert result.path == [0, 1]
