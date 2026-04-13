"""Regression tests for the PPP simulator scene geometry."""

from __future__ import annotations

import os
import sys

# Expose tools/simulator modules (scenes.ppp).
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "..", "src", "arco", "tools", "simulator"),
)

from scenes.ppp import BOUNDS, BOXES, GOAL, START, is_wall


def test_bounds_are_expanded() -> None:
    assert BOUNDS == [(0.0, 60.0), (0.0, 20.0), (0.0, 6.0)]


def test_start_and_goal_use_opposite_large_world_corners() -> None:
    assert START.tolist() == [1.0, 1.0, 0.0]
    assert GOAL.tolist() == [59.0, 19.0, 0.0]


def test_three_crossing_barriers_exist() -> None:
    crossing = [box for box in BOXES if is_wall(box)]
    assert len(crossing) == 4

    full_width = [box for box in crossing if (box[4] - box[1]) >= 19.9]
    half_width = [box for box in crossing if 9.9 <= (box[4] - box[1]) < 19.9]

    assert len(full_width) == 2
    assert len(half_width) == 2


def test_split_third_barrier_has_mixed_heights() -> None:
    split_third = [
        box
        for box in BOXES
        if box[0] == 38.0 and box[3] == 40.0 and (box[4] - box[1]) == 10.0
    ]
    assert len(split_third) == 2

    heights = sorted(box[5] - box[2] for box in split_third)
    assert heights == [1.4, 3.2]
