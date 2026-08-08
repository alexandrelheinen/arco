"""Tests for guidance public API surface."""

from __future__ import annotations

from arco import guidance
from arco.guidance import MovingAverageInterpolator
from arco.guidance.interpolation import MovingAverageInterpolator as MADeep


def test_moving_average_exported_from_guidance():
    assert MovingAverageInterpolator is MADeep
    assert "MovingAverageInterpolator" in guidance.__all__


def test_subpackage_all_exports():
    from arco.mapping import grid
    from arco.planning import continuous, discrete

    assert "ManhattanGrid" in grid.__all__
    assert "RRTPlanner" in continuous.__all__
    assert "AStarPlanner" in discrete.__all__
