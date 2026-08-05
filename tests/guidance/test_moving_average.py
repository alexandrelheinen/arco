"""Tests for MovingAverageInterpolator polyline smoothing."""

from __future__ import annotations

import math

import pytest

from arco.guidance.interpolation import MovingAverageInterpolator


def _wiggle_amplitude(path: list[tuple[float, float]]) -> float:
    """Max mid-region |y| of a nominally straight (y = 0) path.

    The three points nearest each end are excluded: endpoints are
    preserved exactly and pin their neighbors' amplitude.
    """
    return max(abs(p[1]) for p in path[3:-3])


def test_endpoints_are_preserved() -> None:
    path = [(0.0, 0.0), (1.0, 2.0), (2.0, -2.0), (3.0, 0.5), (4.0, 0.0)]
    smoothed = MovingAverageInterpolator(iterations=3).interpolate(path)
    assert smoothed[0] == path[0]
    assert smoothed[-1] == path[-1]
    assert len(smoothed) == len(path)


def test_reduces_lateral_wiggle() -> None:
    # Zigzag around the x-axis: amplitude must shrink monotonically.
    path = [(float(i), (1.0 if i % 2 else -1.0)) for i in range(12)]
    one = MovingAverageInterpolator(iterations=1).interpolate(path)
    three = MovingAverageInterpolator(iterations=3).interpolate(path)
    assert _wiggle_amplitude(one) < _wiggle_amplitude(path)
    assert _wiggle_amplitude(three) < _wiggle_amplitude(one)


def test_straight_line_is_unchanged() -> None:
    path = [(float(i), 0.0) for i in range(8)]
    smoothed = MovingAverageInterpolator(iterations=2).interpolate(path)
    for (x0, y0), (x1, y1) in zip(path, smoothed):
        assert math.isclose(x0, x1, abs_tol=1e-12)
        assert math.isclose(y0, y1, abs_tol=1e-12)


def test_short_paths_pass_through() -> None:
    assert MovingAverageInterpolator().interpolate([]) == []
    assert MovingAverageInterpolator().interpolate([(1.0, 2.0)]) == [
        (1.0, 2.0)
    ]
    two = [(0.0, 0.0), (1.0, 1.0)]
    assert MovingAverageInterpolator().interpolate(two) == two


def test_invalid_window_rejected() -> None:
    with pytest.raises(ValueError):
        MovingAverageInterpolator(window=2)
    with pytest.raises(ValueError):
        MovingAverageInterpolator(window=1)
