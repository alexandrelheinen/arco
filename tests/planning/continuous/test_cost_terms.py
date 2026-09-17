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
