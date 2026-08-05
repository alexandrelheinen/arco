"""Tests for the clearance-guarded reference smoothing in CityScene."""

from __future__ import annotations

import numpy as np
import pytest

pygame = pytest.importorskip("pygame")
pytest.importorskip("OpenGL")

from arco.simulator.scenes.sparse import (  # noqa: E402
    _min_polyline_clearance,
    _smooth_reference,
)


class _PointOccupancy:
    """Minimal occupancy stub with a fixed obstacle point set."""

    def __init__(self, points: list[tuple[float, float]]) -> None:
        self._points = np.asarray(points, dtype=float)

    def nearest_obstacle(self, point: np.ndarray) -> tuple[float, np.ndarray]:
        dists = np.linalg.norm(self._points - point[None, :2], axis=1)
        i = int(np.argmin(dists))
        return float(dists[i]), self._points[i]


def test_smoothing_removes_wiggle_when_clearance_safe() -> None:
    # Wiggly path far away from the single obstacle: smoothing accepted.
    occ = _PointOccupancy([(0.0, 100.0)])
    path = [(float(i), (0.8 if i % 2 else -0.8)) for i in range(12)]
    smoothed = _smooth_reference(path, occ)
    interior = smoothed[1:-1]
    assert max(abs(p[1]) for p in interior) < 0.8
    assert smoothed[0] == path[0]
    assert smoothed[-1] == path[-1]


def test_smoothing_reverted_when_it_cuts_toward_obstacle() -> None:
    # Sharp corner with an obstacle hugging the inside: the moving
    # average would cut toward it, so the guard must return the original.
    occ = _PointOccupancy([(8.5, 1.5)])
    path = [
        (0.0, 0.0),
        (5.0, 0.0),
        (10.0, 0.0),
        (10.0, 5.0),
        (10.0, 10.0),
    ]
    result = _smooth_reference(path, occ)
    assert result == path


def test_min_polyline_clearance_samples_between_waypoints() -> None:
    occ = _PointOccupancy([(5.0, 1.0)])
    # No waypoint is near the obstacle, but the segment midpoint is.
    path = [(0.0, 0.0), (10.0, 0.0)]
    clearance = _min_polyline_clearance(path, occ, sample_spacing=1.0)
    assert clearance == pytest.approx(1.0, abs=0.2)
    # Endpoint-only evaluation would report ≥ 5 m — sampling must see
    # the mid-segment pinch.
    assert clearance < 2.0
