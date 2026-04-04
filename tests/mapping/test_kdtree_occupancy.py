"""Tests for KDTreeOccupancy."""

import numpy as np
import pytest

from arco.mapping import KDTreeOccupancy

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_basic():
    pts = np.array([[1.0, 0.0], [0.0, 1.0]])
    occ = KDTreeOccupancy(pts, clearance=0.5)
    assert occ.dimension == 2
    assert occ.clearance == 0.5


def test_construction_list_input():
    occ = KDTreeOccupancy([[2.0, 3.0], [4.0, 5.0]], clearance=1.0)
    assert occ.points.shape == (2, 2)


def test_construction_empty_raises():
    with pytest.raises(ValueError, match="at least one obstacle"):
        KDTreeOccupancy(np.empty((0, 2)), clearance=0.5)


def test_construction_nonpositive_clearance_raises():
    with pytest.raises(ValueError, match="clearance must be positive"):
        KDTreeOccupancy([[1.0, 1.0]], clearance=0.0)


def test_construction_single_point():
    occ = KDTreeOccupancy([[5.0, 5.0]], clearance=1.0)
    assert occ.dimension == 2


# ---------------------------------------------------------------------------
# nearest_obstacle
# ---------------------------------------------------------------------------


def test_nearest_obstacle_exact():
    occ = KDTreeOccupancy([[1.0, 0.0], [0.0, 1.0]], clearance=0.5)
    dist, pt = occ.nearest_obstacle(np.array([1.0, 0.0]))
    assert dist == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(pt, [1.0, 0.0])


def test_nearest_obstacle_distance():
    occ = KDTreeOccupancy([[3.0, 4.0]], clearance=0.5)
    dist, pt = occ.nearest_obstacle(np.array([0.0, 0.0]))
    assert dist == pytest.approx(5.0)
    assert np.allclose(pt, [3.0, 4.0])


def test_nearest_obstacle_multiple_points():
    occ = KDTreeOccupancy(
        [[0.0, 10.0], [1.0, 1.0], [10.0, 0.0]], clearance=0.5
    )
    dist, pt = occ.nearest_obstacle(np.array([1.1, 1.1]))
    assert dist < 0.2
    assert np.allclose(pt, [1.0, 1.0])


def test_nearest_obstacle_returns_copy():
    occ = KDTreeOccupancy([[2.0, 2.0]], clearance=0.5)
    _, pt = occ.nearest_obstacle(np.array([0.0, 0.0]))
    pt[0] = 999.0
    # Mutating the returned array should not affect the stored data
    _, pt2 = occ.nearest_obstacle(np.array([0.0, 0.0]))
    assert pt2[0] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# is_occupied
# ---------------------------------------------------------------------------


def test_is_occupied_inside_clearance():
    occ = KDTreeOccupancy([[5.0, 5.0]], clearance=2.0)
    assert occ.is_occupied(np.array([5.0, 4.5]))


def test_is_occupied_outside_clearance():
    occ = KDTreeOccupancy([[5.0, 5.0]], clearance=0.5)
    assert not occ.is_occupied(np.array([0.0, 0.0]))


def test_is_occupied_on_boundary():
    # Exactly at clearance distance → NOT occupied (strict <)
    occ = KDTreeOccupancy([[0.0, 0.0]], clearance=1.0)
    assert not occ.is_occupied(np.array([1.0, 0.0]))


def test_is_occupied_3d():
    occ = KDTreeOccupancy([[1.0, 2.0, 3.0]], clearance=0.5)
    assert occ.is_occupied(np.array([1.0, 2.0, 3.0]))
    assert not occ.is_occupied(np.array([10.0, 10.0, 10.0]))


# ---------------------------------------------------------------------------
# points property
# ---------------------------------------------------------------------------


def test_points_property_is_copy():
    pts = np.array([[1.0, 2.0], [3.0, 4.0]])
    occ = KDTreeOccupancy(pts, clearance=0.5)
    copy = occ.points
    copy[0, 0] = 999.0
    assert occ.points[0, 0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# query_distances
# ---------------------------------------------------------------------------


def test_query_distances_batch():
    occ = KDTreeOccupancy([[0.0, 0.0]], clearance=0.5)
    pts = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 0.0]])
    dists = occ.query_distances(pts)
    assert dists.shape == (3,)
    assert dists[0] == pytest.approx(0.0, abs=1e-9)
    assert dists[1] == pytest.approx(5.0)
    assert dists[2] == pytest.approx(1.0)


def test_query_distances_matches_nearest_obstacle():
    occ = KDTreeOccupancy([[2.0, 3.0], [7.0, 8.0]], clearance=0.5)
    pts = np.array([[1.0, 1.0], [6.0, 8.0]])
    dists = occ.query_distances(pts)
    for i, p in enumerate(pts):
        d, _ = occ.nearest_obstacle(p)
        assert dists[i] == pytest.approx(d)
