"""Tests for RRT/SST policy injection hooks."""

from __future__ import annotations

import numpy as np

from arco.mapping.occupancy import Occupancy
from arco.planning import RRTPlanner, SSTPlanner


class _OpenOccupancy(Occupancy):
    def is_occupied(self, position: np.ndarray) -> bool:
        return False

    def nearest_obstacle(self, position: np.ndarray):
        return float("inf"), position


def test_rrt_custom_sampler_is_used():
    occ = _OpenOccupancy()
    calls = {"n": 0}

    def sampler(rng):
        calls["n"] += 1
        return np.array([4.5, 4.5])

    planner = RRTPlanner(
        occ,
        bounds=[(0.0, 5.0), (0.0, 5.0)],
        max_sample_count=50,
        goal_bias=0.0,
        sampler=sampler,
        early_stop=True,
    )
    path = planner.plan(np.array([0.5, 0.5]), np.array([4.5, 4.5]))
    assert path is not None
    assert calls["n"] > 0


def test_rrt_custom_steerer_is_used():
    occ = _OpenOccupancy()
    calls = {"n": 0}

    def steerer(a, b):
        calls["n"] += 1
        return a + 0.5 * (b - a)

    planner = RRTPlanner(
        occ,
        bounds=[(0.0, 5.0), (0.0, 5.0)],
        max_sample_count=200,
        goal_bias=0.5,
        steerer=steerer,
        early_stop=True,
    )
    path = planner.plan(np.array([0.5, 0.5]), np.array([4.5, 4.5]))
    assert path is not None
    assert calls["n"] > 0


def test_rrt_custom_segment_free_can_block_all():
    occ = _OpenOccupancy()

    planner = RRTPlanner(
        occ,
        bounds=[(0.0, 5.0), (0.0, 5.0)],
        max_sample_count=50,
        goal_bias=0.5,
        segment_free=lambda a, b: False,
        early_stop=True,
    )
    path = planner.plan(np.array([0.5, 0.5]), np.array([4.5, 4.5]))
    assert path is None


def test_sst_custom_sampler_is_used():
    occ = _OpenOccupancy()
    calls = {"n": 0}

    def sampler(rng):
        calls["n"] += 1
        return np.array([4.5, 4.5])

    planner = SSTPlanner(
        occ,
        bounds=[(0.0, 5.0), (0.0, 5.0)],
        max_sample_count=80,
        goal_bias=0.0,
        sampler=sampler,
        early_stop=True,
    )
    path = planner.plan(np.array([0.5, 0.5]), np.array([4.5, 4.5]))
    assert path is not None
    assert calls["n"] > 0
