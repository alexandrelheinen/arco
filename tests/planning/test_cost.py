"""Functional tests for PlannerCost defaults and planner inheritance."""

from __future__ import annotations

import numpy as np
import pytest

from arco.mapping import ManhattanGrid
from arco.mapping.occupancy import Occupancy
from arco.planning.continuous import ContinuousPlanner, RRTPlanner, SSTPlanner
from arco.planning.cost import PlannerCost
from arco.planning.discrete import AStarPlanner, DiscretePlanner


class _BareCost(PlannerCost):
    """Minimal subclass used to exercise default cost methods."""


class _OpenOccupancy(Occupancy):
    """Occupancy map with no obstacles."""

    def is_occupied(self, position: np.ndarray) -> bool:
        return False

    def nearest_obstacle(self, position: np.ndarray):
        return float("inf"), position


def test_planner_cost_default_distance_is_euclidean():
    cost = _BareCost()
    assert cost.distance([0.0, 0.0], [3.0, 4.0]) == pytest.approx(5.0)


def test_planner_cost_default_heuristic_matches_distance():
    cost = _BareCost()
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 6.0, 3.0])
    assert cost.heuristic(a, b) == pytest.approx(cost.distance(a, b))


def test_discrete_and_continuous_bases_inherit_planner_cost():
    assert issubclass(DiscretePlanner, PlannerCost)
    assert issubclass(ContinuousPlanner, PlannerCost)


def test_astar_rrt_sst_inherit_planner_cost():
    assert issubclass(AStarPlanner, PlannerCost)
    assert issubclass(RRTPlanner, PlannerCost)
    assert issubclass(SSTPlanner, PlannerCost)


def test_astar_uses_overridable_distance_and_heuristic():
    grid = ManhattanGrid((4, 4))

    class CountingAStar(AStarPlanner):
        def __init__(self, graph):
            super().__init__(graph)
            self.distance_calls = 0
            self.heuristic_calls = 0

        def distance(self, state_a, state_b) -> float:
            self.distance_calls += 1
            return super().distance(state_a, state_b)

        def heuristic(self, state_a, state_b) -> float:
            self.heuristic_calls += 1
            return super().heuristic(state_a, state_b)

    planner = CountingAStar(grid)
    path = planner.plan((0, 0), (3, 3))
    assert path is not None
    assert planner.distance_calls > 0
    assert planner.heuristic_calls > 0


def test_astar_custom_heuristic_callable_still_works():
    grid = ManhattanGrid((5, 5))
    calls = {"n": 0}

    def zero_heuristic(node, goal) -> float:
        calls["n"] += 1
        return 0.0

    planner = AStarPlanner(grid, heuristic=zero_heuristic)
    path = planner.plan((0, 0), (4, 4))
    assert path is not None
    assert calls["n"] > 0


def test_continuous_distance_uses_step_size_normalization():
    occ = _OpenOccupancy()
    planner = RRTPlanner(
        occ,
        bounds=[(0.0, 10.0), (0.0, 10.0)],
        step_size=np.array([2.0, 1.0]),
        max_sample_count=1,
    )
    a = np.array([0.0, 0.0])
    b = np.array([2.0, 0.0])
    # Normalized delta = (1, 0) → distance 1.
    assert planner.distance(a, b) == pytest.approx(1.0)
    assert planner.heuristic(a, b) == pytest.approx(1.0)


def test_rrt_uses_overridable_distance():
    occ = _OpenOccupancy()

    class CountingRRT(RRTPlanner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.distance_calls = 0

        def distance(self, state_a, state_b) -> float:
            self.distance_calls += 1
            return super().distance(state_a, state_b)

    planner = CountingRRT(
        occ,
        bounds=[(0.0, 5.0), (0.0, 5.0)],
        step_size=1.0,
        max_sample_count=200,
        goal_bias=0.3,
        early_stop=True,
    )
    path = planner.plan(np.array([0.5, 0.5]), np.array([4.5, 4.5]))
    assert path is not None
    assert planner.distance_calls > 0


def test_sst_uses_overridable_distance():
    occ = _OpenOccupancy()

    class CountingSST(SSTPlanner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.distance_calls = 0

        def distance(self, state_a, state_b) -> float:
            self.distance_calls += 1
            return super().distance(state_a, state_b)

    planner = CountingSST(
        occ,
        bounds=[(0.0, 5.0), (0.0, 5.0)],
        step_size=1.0,
        max_sample_count=300,
        goal_bias=0.3,
        early_stop=True,
    )
    path = planner.plan(np.array([0.5, 0.5]), np.array([4.5, 4.5]))
    assert path is not None
    assert planner.distance_calls > 0
