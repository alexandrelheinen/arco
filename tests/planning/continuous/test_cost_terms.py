"""Tests for TrajectoryOptimizer cost_terms injection (Tier C2)."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from arco.mapping import KDTreeOccupancy
from arco.planning.continuous import TrajectoryOptimizer
from arco.planning.continuous.cost_terms import (
    CollisionCostTerm,
    DeviationCostTerm,
    DynamicsCostTerm,
    TimeCostTerm,
    VelocityCostTerm,
    build_default_cost_terms,
)
from arco.protocols import CostTerm


def _free_occupancy(clearance=0.3):
    return KDTreeOccupancy([[200.0, 200.0]], clearance=clearance)


class _ConstantTerm:
    """Minimal CostTerm that adds a fixed offset."""

    name = "constant"

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def __call__(self, context: Dict[str, Any]) -> float:
        return self.value


def test_default_cost_terms_has_five_named_terms():
    terms = build_default_cost_terms(
        weight_time=1.0,
        weight_deviation=1.0,
        weight_velocity=1.0,
        weight_collision=1.0,
        weight_dynamics=1.0,
        cruise_speed=1.0,
        collision_barrier_scale=50.0,
        collision_barrier_power=4.0,
        max_speed=None,
        min_speed=None,
    )
    assert len(terms) == 5
    assert [t.name for t in terms] == [
        "time",
        "deviation",
        "velocity",
        "collision",
        "dynamics",
    ]
    assert all(isinstance(t, CostTerm) for t in terms)


def test_optimizer_builds_five_default_terms():
    occ = _free_occupancy()
    opt = TrajectoryOptimizer(occ, cruise_speed=1.0)
    assert len(opt.cost_terms) == 5
    assert [t.name for t in opt.cost_terms] == [
        "time",
        "deviation",
        "velocity",
        "collision",
        "dynamics",
    ]


def test_default_terms_match_analytical_time_only():
    """Default time term must match w_time · T² exactly."""
    occ = _free_occupancy()
    opt = TrajectoryOptimizer(
        occ,
        weight_time=3.0,
        weight_deviation=0.0,
        weight_velocity=0.0,
        weight_collision=0.0,
        weight_dynamics=0.0,
        collision_barrier_scale=0.0,
    )
    ref = [np.array([0.0, 0.0]), np.array([4.0, 0.0])]
    x = np.array([2.5])
    # T = 2.5 → J = 3 * 2.5² = 18.75
    assert opt._cost(x, ref, 1, 2) == 3.0 * 2.5**2


def test_custom_cost_term_injection_replaces_defaults():
    occ = _free_occupancy()
    custom: List[CostTerm] = [_ConstantTerm(7.5)]
    opt = TrajectoryOptimizer(occ, cost_terms=custom)
    assert len(opt.cost_terms) == 1
    ref = [np.array([0.0, 0.0]), np.array([1.0, 0.0])]
    x = np.array([1.0])
    assert opt._cost(x, ref, 1, 2) == 7.5


def test_custom_term_appended_to_defaults_increases_cost():
    occ = _free_occupancy()
    base = TrajectoryOptimizer(
        occ,
        weight_time=1.0,
        weight_deviation=0.0,
        weight_velocity=0.0,
        weight_collision=0.0,
        weight_dynamics=0.0,
        collision_barrier_scale=0.0,
    )
    augmented_terms = list(base.cost_terms) + [_ConstantTerm(5.0)]
    aug = TrajectoryOptimizer(
        occ,
        weight_time=1.0,
        weight_deviation=0.0,
        weight_velocity=0.0,
        weight_collision=0.0,
        weight_dynamics=0.0,
        collision_barrier_scale=0.0,
        cost_terms=augmented_terms,
    )
    ref = [np.array([0.0, 0.0]), np.array([4.0, 0.0])]
    x = np.array([2.0])
    assert aug._cost(x, ref, 1, 2) == base._cost(x, ref, 1, 2) + 5.0


def test_individual_default_terms_evaluate():
    occ = _free_occupancy()
    ref = [
        np.array([0.0, 0.0]),
        np.array([4.0, 1.0]),
        np.array([8.0, 0.0]),
    ]
    opt = TrajectoryOptimizer(
        occ,
        cruise_speed=2.0,
        weight_time=2.0,
        weight_deviation=3.0,
        weight_velocity=4.0,
        weight_collision=0.0,
        weight_dynamics=0.0,
        collision_barrier_scale=0.0,
        sample_count=0,
    )
    # durations 1,1; interior at (4, 3) vs ref (4, 1)
    x = np.array([1.0, 1.0, 4.0, 3.0])
    durations, waypoints = opt._unpack(x, ref, 2, 2)
    pts = np.array(waypoints)
    durs = np.maximum(durations, 1e-9)
    lengths = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    speeds = lengths / durs
    context = {
        "durs": durs,
        "pts": pts,
        "speeds": speeds,
        "ref": ref,
        "segment_count": 2,
        "occupancy": occ,
        "sample_count": 0,
    }

    assert TimeCostTerm(2.0)(context) == 2.0 * (1.0 + 1.0) ** 2
    assert DeviationCostTerm(3.0)(context) == 3.0 * (3.0 - 1.0) ** 2
    expected_vel = 4.0 * float(np.sum((speeds - 2.0) ** 2))
    assert VelocityCostTerm(4.0, 2.0)(context) == expected_vel
    assert CollisionCostTerm(0.0)(context) == 0.0
    assert DynamicsCostTerm(0.0)(context) == 0.0


def test_defaults_still_optimize():
    occ = _free_occupancy()
    opt = TrajectoryOptimizer(
        occ,
        cruise_speed=2.0,
        weight_time=10.0,
        max_iter=20,
    )
    ref = [np.array([0.0, 5.0]), np.array([4.0, 5.0]), np.array([8.0, 5.0])]
    result = opt.optimize(ref)
    assert len(result.states) == 3
    assert len(result.durations) == 2
    assert np.isfinite(result.cost)
