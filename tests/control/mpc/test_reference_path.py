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


def test_reference_path_curvature_detects_sharp_l_kink() -> None:
    """A 90° polyline corner must produce usable κ for ``v_curve`` braking.

    The previous skip-one finite difference reported κ≈0 at the vertex of
    ``(0,0)→(50,0)→(50,50)`` because ``h[i-1]`` and ``h[i+1]`` cancelled.
    """
    path = ReferencePath([(0.0, 0.0), (50.0, 0.0), (50.0, 50.0)])
    kappa_corner = abs(path.curvature(50.0))
    # Spread over ds_cap=20 m → |κ| = (π/2) / 20 ≈ 0.0785.
    assert kappa_corner > 0.05
    # Approach preview must expose the corner κ well before the vertex so
    # the NMPC can decelerate (city soft a_max needs tens of meters).
    kappa_approach = abs(path.curvature(20.0))
    assert kappa_approach > 0.05


def test_reference_path_curvature_grid_corner_limits_city_speed() -> None:
    """A* 15 m grid corners yield κ that brakes soft city ω to lane-safe v."""
    path = ReferencePath(
        [
            (0.0, 0.0),
            (15.0, 0.0),
            (30.0, 0.0),
            (45.0, 0.0),
            (60.0, 0.0),
            (60.0, 15.0),
            (60.0, 30.0),
        ]
    )
    kappa = abs(path.curvature(60.0))
    omega = math.radians(40.0)
    v_curve = omega / max(kappa, 1e-6)
    # R = v/ω = 1/κ; with consecutive turn / 15 m, R ≈ 9.5 m < 15 m lane.
    assert v_curve / omega < 15.0
    assert v_curve < 12.0


def test_reference_path_total_length() -> None:
    path = ReferencePath([(0.0, 0.0), (3.0, 4.0)])
    assert abs(path.total_length - 5.0) < 1e-9
    x, y = path.position(2.5)
    assert abs(x - 1.5) < 1e-9
    assert abs(y - 2.0) < 1e-9
