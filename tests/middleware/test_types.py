"""Tests for MappingFrame, PlanFrame, and GuidanceFrame dataclasses."""

from __future__ import annotations

import time

from arco.middleware.types import GuidanceFrame, MappingFrame, PlanFrame


# ---------------------------------------------------------------------------
# MappingFrame
# ---------------------------------------------------------------------------


def test_mapping_frame_stores_timestamp():
    ts = time.monotonic()
    frame = MappingFrame(timestamp=ts)
    assert frame.timestamp == ts


def test_mapping_frame_default_fields():
    frame = MappingFrame(timestamp=0.0)
    assert frame.obstacle_points == []
    assert len(frame.bounds) == 4
    assert frame.clearance == 0.0


def test_mapping_frame_custom_fields():
    pts = [[1.0, 2.0], [3.0, 4.0]]
    bounds = [0.0, 0.0, 10.0, 10.0]
    frame = MappingFrame(
        timestamp=1.0,
        obstacle_points=pts,
        bounds=bounds,
        clearance=0.5,
    )
    assert frame.obstacle_points == pts
    assert frame.bounds == bounds
    assert frame.clearance == 0.5


def test_mapping_frame_default_bounds_are_independent():
    # Each instance must have its own list, not a shared default.
    f1 = MappingFrame(timestamp=0.0)
    f2 = MappingFrame(timestamp=0.0)
    f1.bounds[0] = 99.0
    assert f2.bounds[0] == 0.0


# ---------------------------------------------------------------------------
# PlanFrame
# ---------------------------------------------------------------------------


def test_plan_frame_stores_timestamp():
    ts = time.monotonic()
    frame = PlanFrame(timestamp=ts)
    assert frame.timestamp == ts


def test_plan_frame_default_fields():
    frame = PlanFrame(timestamp=0.0)
    assert frame.waypoints == []
    assert frame.planner == ""


def test_plan_frame_custom_fields():
    wps = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
    frame = PlanFrame(timestamp=2.0, waypoints=wps, planner="RRT*")
    assert frame.waypoints == wps
    assert frame.planner == "RRT*"


def test_plan_frame_waypoints_are_independent():
    f1 = PlanFrame(timestamp=0.0)
    f2 = PlanFrame(timestamp=0.0)
    f1.waypoints.append([1.0, 1.0])
    assert f2.waypoints == []


# ---------------------------------------------------------------------------
# GuidanceFrame
# ---------------------------------------------------------------------------


def test_guidance_frame_stores_timestamp():
    ts = time.monotonic()
    frame = GuidanceFrame(timestamp=ts)
    assert frame.timestamp == ts


def test_guidance_frame_default_fields():
    frame = GuidanceFrame(timestamp=0.0)
    assert frame.trajectory == []
    assert frame.durations == []


def test_guidance_frame_custom_fields():
    traj = [[0.0, 0.0], [1.0, 1.0]]
    durs = [0.5]
    frame = GuidanceFrame(timestamp=3.0, trajectory=traj, durations=durs)
    assert frame.trajectory == traj
    assert frame.durations == durs


def test_guidance_frame_lists_are_independent():
    f1 = GuidanceFrame(timestamp=0.0)
    f2 = GuidanceFrame(timestamp=0.0)
    f1.trajectory.append([5.0, 5.0])
    assert f2.trajectory == []
