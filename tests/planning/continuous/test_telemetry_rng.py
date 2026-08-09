"""Tests for telemetry publisher and RNG injection."""

from __future__ import annotations

import numpy as np

from arco.mapping.occupancy import Occupancy
from arco.planning import RRTPlanner
from arco.planning.continuous.telemetry import noop_publisher


class _OpenOccupancy(Occupancy):
    def is_occupied(self, position: np.ndarray) -> bool:
        return False

    def nearest_obstacle(self, position: np.ndarray):
        return float("inf"), position


def test_custom_publisher_receives_snapshots():
    occ = _OpenOccupancy()
    snapshots = []

    planner = RRTPlanner(
        occ,
        bounds=[(0.0, 5.0), (0.0, 5.0)],
        max_sample_count=150,
        goal_bias=0.5,
        publisher=snapshots.append,
        seed=0,
        early_stop=True,
    )
    path = planner.plan(np.array([0.5, 0.5]), np.array([4.5, 4.5]))
    assert path is not None
    assert len(snapshots) >= 1


def test_noop_publisher_disables_default_ipc():
    occ = _OpenOccupancy()
    planner = RRTPlanner(
        occ,
        bounds=[(0.0, 2.0), (0.0, 2.0)],
        max_sample_count=20,
        publisher=noop_publisher,
        seed=1,
    )
    # Should not raise even if temp file is unavailable.
    planner.plan(np.array([0.1, 0.1]), np.array([1.9, 1.9]))


def test_seed_makes_sampling_deterministic():
    occ = _OpenOccupancy()
    kwargs = dict(
        occupancy=occ,
        bounds=[(0.0, 5.0), (0.0, 5.0)],
        max_sample_count=80,
        goal_bias=0.1,
        publisher=noop_publisher,
        seed=42,
        early_stop=False,
    )
    p1 = RRTPlanner(**kwargs).plan(np.array([0.5, 0.5]), np.array([4.0, 4.0]))
    p2 = RRTPlanner(**kwargs).plan(np.array([0.5, 0.5]), np.array([4.0, 4.0]))
    assert p1 is not None and p2 is not None
    assert len(p1) == len(p2)
    for a, b in zip(p1, p2):
        assert np.allclose(a, b)
