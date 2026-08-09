"""Unit tests for shared MPC soft-barrier cost helpers."""

from __future__ import annotations

import math

import pytest

from arco.control.mpc.costs import forward_cone_factor, obstacle_barrier


def _ref_obstacle_barrier(
    distance: float,
    clearance: float,
    weight: float,
    power: float,
    cone_factor: float,
) -> float:
    """Float reference matching the former inlined MPC barrier."""
    denom = max(clearance, 1e-6)
    penetration = max(0.0, (clearance - distance) / denom)
    directional = 0.2 + 0.8 * cone_factor
    return weight * (penetration**power) * directional


def _ref_forward_cone(
    pose_x: float,
    pose_y: float,
    heading: float,
    obstacle_x: float,
    obstacle_y: float,
) -> float:
    """Float reference: heading projection (path-following formula)."""
    dx = obstacle_x - pose_x
    dy = obstacle_y - pose_y
    dist = math.sqrt(dx * dx + dy * dy + 1e-9)
    fwd = (math.cos(heading) * dx + math.sin(heading) * dy) / dist
    return max(0.0, fwd)


@pytest.mark.parametrize(
    "distance,clearance,weight,power,cone_factor",
    [
        (0.5, 1.0, 10.0, 4.0, 1.0),
        (0.2, 1.0, 10.0, 4.0, 0.0),
        (0.0, 0.5, 5.0, 2.0, 0.5),
        (2.0, 1.0, 10.0, 4.0, 1.0),  # outside clearance → 0
        (0.8, 1.0, 1.0, 4.0, 1.0),  # joint-space: cone_factor=1
    ],
)
def test_obstacle_barrier_matches_float_reference(
    distance: float,
    clearance: float,
    weight: float,
    power: float,
    cone_factor: float,
) -> None:
    expected = _ref_obstacle_barrier(
        distance, clearance, weight, power, cone_factor
    )
    got = obstacle_barrier(distance, clearance, weight, power, cone_factor)
    assert got == pytest.approx(expected, rel=0.0, abs=1e-12)


@pytest.mark.parametrize(
    "pose_x,pose_y,heading,ox,oy",
    [
        (0.0, 0.0, 0.0, 1.0, 0.0),  # straight ahead → 1
        (0.0, 0.0, 0.0, -1.0, 0.0),  # behind → 0
        (0.0, 0.0, 0.0, 0.0, 1.0),  # 90° → 0
        (1.0, 2.0, math.pi / 4, 2.0, 3.0),
    ],
)
def test_forward_cone_factor_matches_float_reference(
    pose_x: float,
    pose_y: float,
    heading: float,
    ox: float,
    oy: float,
) -> None:
    expected = _ref_forward_cone(pose_x, pose_y, heading, ox, oy)
    got = forward_cone_factor(pose_x, pose_y, heading, ox, oy)
    assert got == pytest.approx(expected, rel=0.0, abs=1e-12)
    # Also matches cos(heading − bearing) away from the origin.
    bearing = math.atan2(oy - pose_y, ox - pose_x)
    atan2_ref = max(0.0, math.cos(heading - bearing))
    assert got == pytest.approx(atan2_ref, abs=1e-9)


def test_joint_space_cone_factor_one_is_identity_directional() -> None:
    """cone_factor=1.0 → directional weight is exactly 1 (joint-space)."""
    distance, clearance, weight, power = 0.25, 1.0, 8.0, 4.0
    got = obstacle_barrier(distance, clearance, weight, power, 1.0)
    denom = max(clearance, 1e-6)
    penetration = max(0.0, (clearance - distance) / denom)
    assert got == pytest.approx(weight * (penetration**power))


def test_casadi_obstacle_barrier_matches_float() -> None:
    ca = pytest.importorskip("casadi")
    distance, clearance = 0.4, 1.0
    weight, power, cone = 10.0, 4.0, 0.75
    expected = _ref_obstacle_barrier(distance, clearance, weight, power, cone)
    d = ca.MX.sym("d")
    c = ca.MX.sym("c")
    cf = ca.MX.sym("cf")
    expr = obstacle_barrier(d, c, weight, power, cf)
    fn = ca.Function("b", [d, c, cf], [expr])
    got = float(fn(distance, clearance, cone))
    assert got == pytest.approx(expected, abs=1e-12)


def test_casadi_forward_cone_matches_float() -> None:
    ca = pytest.importorskip("casadi")
    pose_x, pose_y, heading = 0.0, 0.0, 0.3
    ox, oy = 1.5, 0.4
    expected = _ref_forward_cone(pose_x, pose_y, heading, ox, oy)
    px = ca.MX.sym("px")
    py = ca.MX.sym("py")
    th = ca.MX.sym("th")
    ox_s = ca.MX.sym("ox")
    oy_s = ca.MX.sym("oy")
    expr = forward_cone_factor(px, py, th, ox_s, oy_s)
    fn = ca.Function("cone", [px, py, th, ox_s, oy_s], [expr])
    got = float(fn(pose_x, pose_y, heading, ox, oy))
    assert got == pytest.approx(expected, abs=1e-12)
