"""Tests for DubinsPrimitive: turning-radius dynamic constraint."""

from __future__ import annotations

import numpy as np
import pytest

from arco.guidance.primitive.dubins import DubinsPrimitive

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_stores_turning_radius() -> None:
    prim = DubinsPrimitive(turning_radius=2.5)
    assert prim.turning_radius == 2.5


# ---------------------------------------------------------------------------
# is_feasible — state-length semantics
# ---------------------------------------------------------------------------


def test_is_feasible_2d_state_always_true() -> None:
    """A 2-element position state carries no curvature info → always True."""
    prim = DubinsPrimitive(turning_radius=1.0)
    assert prim.is_feasible(np.array([3.0, 4.0])) is True


def test_is_feasible_3d_state_always_true() -> None:
    """A 3-element (x, y, θ) state carries no speed/turn-rate → always True."""
    prim = DubinsPrimitive(turning_radius=1.0)
    assert prim.is_feasible(np.array([0.0, 0.0, 1.57])) is True


def test_is_feasible_4d_state_always_true() -> None:
    """A 4-element state includes speed but no turn rate → always True."""
    prim = DubinsPrimitive(turning_radius=1.0)
    assert prim.is_feasible(np.array([0.0, 0.0, 0.0, 5.0])) is True


# ---------------------------------------------------------------------------
# is_feasible — 5-element state (x, y, θ, v, ω)
# ---------------------------------------------------------------------------


def test_is_feasible_straight_line_always_true() -> None:
    """Zero turn rate (straight-line motion) is always feasible."""
    prim = DubinsPrimitive(turning_radius=1.0)
    state = np.array([0.0, 0.0, 0.0, 3.0, 0.0])
    assert prim.is_feasible(state) is True


def test_is_feasible_large_radius_feasible() -> None:
    """Turning radius |v/ω| > min_radius → feasible."""
    prim = DubinsPrimitive(turning_radius=1.0)
    # v=4, ω=1 → radius=4 > 1.0 ✓
    state = np.array([0.0, 0.0, 0.0, 4.0, 1.0])
    assert prim.is_feasible(state) is True


def test_is_feasible_exact_minimum_radius_feasible() -> None:
    """Turning radius equal to min_radius is feasible (boundary case)."""
    prim = DubinsPrimitive(turning_radius=2.0)
    # v=4, ω=2 → radius=2.0 == turning_radius ✓
    state = np.array([0.0, 0.0, 0.0, 4.0, 2.0])
    assert prim.is_feasible(state) is True


def test_is_feasible_radius_too_small_rejected() -> None:
    """Turning radius |v/ω| < min_radius → infeasible."""
    prim = DubinsPrimitive(turning_radius=2.0)
    # v=1, ω=2 → radius=0.5 < 2.0 ✗
    state = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
    assert prim.is_feasible(state) is False


def test_is_feasible_negative_turn_rate_feasible() -> None:
    """Negative turn rate uses |ω| for radius computation."""
    prim = DubinsPrimitive(turning_radius=1.0)
    # v=4, ω=-1 → |radius|=4 > 1.0 ✓
    state = np.array([0.0, 0.0, 0.0, 4.0, -1.0])
    assert prim.is_feasible(state) is True


def test_is_feasible_negative_turn_rate_rejected() -> None:
    """Negative turn rate that violates min_radius is rejected."""
    prim = DubinsPrimitive(turning_radius=2.0)
    # v=1, ω=-4 → |radius|=0.25 < 2.0 ✗
    state = np.array([0.0, 0.0, 0.0, 1.0, -4.0])
    assert prim.is_feasible(state) is False
