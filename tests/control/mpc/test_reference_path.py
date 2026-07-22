"""Tests for ReferencePath arc-length queries."""

from __future__ import annotations

import math

from arco.control.mpc.reference_path import ReferencePath


def test_reference_path_project_straight_line() -> None:
    path = ReferencePath([(0.0, 0.0), (10.0, 0.0)])
    s, lat, head = path.project((5.0, 0.0, 0.0))
    assert abs(s - 5.0) < 1e-9
    assert abs(lat) < 1e-9
    assert abs(head) < 1e-9


def test_reference_path_project_lateral_offset() -> None:
    path = ReferencePath([(0.0, 0.0), (10.0, 0.0)])
    s, lat, head = path.project((3.0, 0.4, 0.0))
    assert abs(s - 3.0) < 1e-6
    assert abs(lat - 0.4) < 1e-6
    assert abs(head) < 1e-6


def test_reference_path_project_window_rejects_distant_nearest() -> None:
    """Local window keeps contouring progress on the active approach lane.

    A hairpin makes a pose just past the first corner geometrically closer to
    the return leg; without a window the global nearest point flips forward
    by nearly the full length — the failure mode behind city A* junction
    loops.
    """
    path = ReferencePath(
        [
            (0.0, 0.0),
            (20.0, 0.0),
            (20.0, 0.5),
            (0.0, 0.5),
        ]
    )
    # Past the corner on the outbound lane, closer to the inbound return.
    pose = (10.0, 0.35, 0.0)
    s_global, _, _ = path.project(pose)
    assert s_global > 25.0
    s_local, _, _ = path.project(pose, s_hint=10.0, window=8.0)
    assert s_local < 15.0
    assert abs(s_local - 10.0) < 1e-6


def test_reference_path_tangent_and_curvature_straight() -> None:
    path = ReferencePath([(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)])
    tx, ty = path.tangent(4.0)
    assert abs(tx - 1.0) < 1e-9
    assert abs(ty) < 1e-9
    assert abs(path.curvature(4.0)) < 1e-6
    assert abs(path.heading(4.0)) < 1e-9


def test_reference_path_total_length() -> None:
    path = ReferencePath([(0.0, 0.0), (3.0, 4.0)])
    assert abs(path.total_length - 5.0) < 1e-9
    x, y = path.position(2.5)
    assert abs(x - 1.5) < 1e-9
    assert abs(y - 2.0) < 1e-9
