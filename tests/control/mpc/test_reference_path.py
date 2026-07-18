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
