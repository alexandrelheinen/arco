"""Tests for continuous planner cost= composition."""

from __future__ import annotations

import numpy as np

from arco.mapping.occupancy import Occupancy
from arco.planning import RRTPlanner
from arco.planning.cost import PlannerCost


class _OpenOccupancy(Occupancy):
    def is_occupied(self, position: np.ndarray) -> bool:
        return False

    def nearest_obstacle(self, position: np.ndarray):
        return float("inf"), position


class _ScaledCost(PlannerCost):
    def distance(self, state_a, state_b) -> float:
        return 2.0 * super().distance(state_a, state_b)


def test_cost_injection_delegates_distance():
    occ = _OpenOccupancy()
    cost = _ScaledCost()
    planner = RRTPlanner(
        occ,
        bounds=[(0.0, 5.0), (0.0, 5.0)],
        step_size=1.0,
        max_sample_count=1,
        cost=cost,
    )
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    assert planner.distance(a, b) == 10.0
    assert planner.heuristic(a, b) == 10.0


def test_default_cost_still_step_normalized():
    occ = _OpenOccupancy()
    planner = RRTPlanner(
        occ,
        bounds=[(0.0, 5.0), (0.0, 5.0)],
        step_size=np.array([2.0, 1.0]),
        max_sample_count=1,
    )
    assert planner.distance(np.array([0.0, 0.0]), np.array([2.0, 0.0])) == 1.0
