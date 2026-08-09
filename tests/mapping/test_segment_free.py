"""Tests for Occupancy.segment_free shared collision checks."""

from __future__ import annotations

import numpy as np

from arco.mapping.kdtree import KDTreeOccupancy
from arco.mapping.occupancy import Occupancy
from arco.planning import RRTPlanner


class _WallOccupancy(Occupancy):
    def nearest_obstacle(self, point: np.ndarray):
        return abs(float(point[0]) - 0.5), np.array([0.5, point[1]])

    def is_occupied(self, point: np.ndarray) -> bool:
        return abs(float(point[0]) - 0.5) < 0.05


def test_occupancy_segment_free_default_blocks_wall():
    occ = _WallOccupancy()
    assert occ.segment_free(np.array([0.0, 0.0]), np.array([0.0, 1.0]))
    assert not occ.segment_free(np.array([0.0, 0.0]), np.array([1.0, 0.0]))


def test_occupancy_default_clearance_and_query_distances():
    occ = _WallOccupancy()
    assert occ.clearance == 0.0
    dists = occ.query_distances(np.array([[0.0, 0.0], [0.5, 0.0]]))
    assert dists.shape == (2,)
    assert dists[1] < dists[0]


def test_rrt_uses_occupancy_segment_free():
    occ = KDTreeOccupancy(np.array([[10.0, 10.0]]), clearance=0.1)
    planner = RRTPlanner(
        occ, bounds=[(0.0, 1.0), (0.0, 1.0)], max_sample_count=1
    )
    assert planner.is_segment_free(
        np.array([0.0, 0.0]), np.array([0.5, 0.5])
    )
