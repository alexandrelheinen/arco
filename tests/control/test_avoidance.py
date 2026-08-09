"""Tests for ArtificialPotentialField and TrackingLoop avoidance injection."""

from __future__ import annotations

import numpy as np

from arco.control import ArtificialPotentialField, TrackingLoop
from arco.control.avoidance import ArtificialPotentialField as APFDirect
from arco.control.pure_pursuit import PurePursuitController
from arco.guidance.vehicle import DubinsVehicle
from arco.protocols.avoidance import AvoidanceStrategy

STRAIGHT_PATH: list[tuple[float, float]] = [(float(i), 0.0) for i in range(40)]


class _NearObstacleOccupancy:
    """Minimal occupancy stub: single obstacle at a fixed position."""

    def __init__(
        self,
        obs_x: float,
        obs_y: float,
        clearance: float,
    ) -> None:
        self.clearance = clearance
        self._obs = (obs_x, obs_y)

    def nearest_obstacle(
        self, point: np.ndarray
    ) -> tuple[float, np.ndarray]:
        obs = np.array(self._obs, dtype=float)
        dist = float(np.linalg.norm(np.asarray(point, dtype=float) - obs))
        return dist, obs


def _make_vehicle() -> DubinsVehicle:
    """Create a DubinsVehicle with relaxed actuator limits for unit tests."""
    return DubinsVehicle(
        x=0.0,
        y=0.0,
        heading=0.0,
        max_speed=5.0,
        min_speed=0.0,
        max_turn_rate=4.0,
        max_acceleration=10.0,
        max_turn_rate_dot=10.0,
    )


def _make_apf(
    obs_x: float,
    obs_y: float,
    clearance: float = 2.0,
    repulsion_gain: float = 1.5,
) -> ArtificialPotentialField:
    """Build an APF aimed at a single fixed obstacle."""
    occ = _NearObstacleOccupancy(obs_x, obs_y, clearance)
    return ArtificialPotentialField(
        occupancy=occ, repulsion_gain=repulsion_gain
    )


# ---------------------------------------------------------------------------
# ArtificialPotentialField unit behaviour
# ---------------------------------------------------------------------------


def test_apf_exported_from_control_package() -> None:
    """ArtificialPotentialField is re-exported from arco.control."""
    assert ArtificialPotentialField is APFDirect


def test_apf_satisfies_avoidance_strategy_protocol() -> None:
    """APF instances are runtime-checkable AvoidanceStrategy objects."""
    apf = ArtificialPotentialField()
    assert isinstance(apf, AvoidanceStrategy)


def test_apf_noop_when_occupancy_none() -> None:
    """Disabled path: no occupancy → always zero correction."""
    apf = ArtificialPotentialField(occupancy=None, repulsion_gain=2.0)
    assert apf(0.0, 0.0, 0.0) == 0.0


def test_apf_noop_when_gain_nonpositive() -> None:
    """Disabled path: non-positive gain → always zero correction."""
    occ = _NearObstacleOccupancy(0.5, 0.0, clearance=2.0)
    apf = ArtificialPotentialField(occupancy=occ, repulsion_gain=0.0)
    assert apf(0.0, 0.0, 0.0) == 0.0
    apf_neg = ArtificialPotentialField(occupancy=occ, repulsion_gain=-1.0)
    assert apf_neg(0.0, 0.0, 0.0) == 0.0


def test_apf_zero_outside_influence_radius() -> None:
    """No repulsion when obstacle is beyond 2 × clearance."""
    apf = _make_apf(obs_x=100.0, obs_y=0.0, clearance=2.0)
    assert apf(0.0, 0.0, 0.0) == 0.0


def test_apf_nonzero_inside_influence_radius() -> None:
    """Repulsion is non-zero when obstacle is within 2 × clearance."""
    apf = _make_apf(obs_x=1.5, obs_y=0.0, clearance=2.0)
    assert apf(0.0, 0.0, 0.0) != 0.0


def test_apf_direction_obstacle_left() -> None:
    """Obstacle to the left → negative turn rate (steer right)."""
    apf = _make_apf(obs_x=1.0, obs_y=1.0, clearance=4.0)
    assert apf(0.0, 0.0, 0.0) < 0.0


def test_apf_direction_obstacle_right() -> None:
    """Obstacle to the right → positive turn rate (steer left)."""
    apf = _make_apf(obs_x=1.0, obs_y=-1.0, clearance=4.0)
    assert apf(0.0, 0.0, 0.0) > 0.0


def test_apf_magnitude_scales_with_gain() -> None:
    """Doubling repulsion_gain doubles the correction magnitude."""
    occ = _NearObstacleOccupancy(1.0, 1.0, clearance=4.0)
    r1 = ArtificialPotentialField(occ, repulsion_gain=1.0)(0.0, 0.0, 0.0)
    r2 = ArtificialPotentialField(occ, repulsion_gain=2.0)(0.0, 0.0, 0.0)
    assert abs(r2) == abs(2.0 * r1)


# ---------------------------------------------------------------------------
# TrackingLoop default APF (backwards-compatible kwargs)
# ---------------------------------------------------------------------------


def test_tracking_loop_default_apf_from_occupancy_kwargs() -> None:
    """When avoidance is omitted, occupancy + gain build a default APF."""
    occ = _NearObstacleOccupancy(1.0, 1.0, clearance=4.0)
    loop = TrackingLoop(
        _make_vehicle(),
        PurePursuitController(lookahead_distance=2.0),
        cruise_speed=1.0,
        occupancy=occ,
        repulsion_gain=1.5,
    )
    assert loop._avoidance is not None
    assert isinstance(loop._avoidance, ArtificialPotentialField)
    r = loop._repulsion_turn_rate(0.0, 0.0, 0.0)
    assert r < 0.0


def test_tracking_loop_no_default_apf_when_gain_zero() -> None:
    """repulsion_gain=0 leaves avoidance disabled even with occupancy."""
    occ = _NearObstacleOccupancy(0.5, 0.0, clearance=2.0)
    loop = TrackingLoop(
        _make_vehicle(),
        PurePursuitController(lookahead_distance=2.0),
        cruise_speed=1.0,
        occupancy=occ,
        repulsion_gain=0.0,
    )
    assert loop._avoidance is None
    assert loop._repulsion_turn_rate(0.0, 0.0, 0.0) == 0.0


def test_tracking_loop_no_default_apf_without_occupancy() -> None:
    """No occupancy → no default avoidance, even with positive gain."""
    loop = TrackingLoop(
        _make_vehicle(),
        PurePursuitController(lookahead_distance=2.0),
        cruise_speed=1.0,
        repulsion_gain=2.0,
    )
    assert loop._avoidance is None
    assert loop._repulsion_turn_rate(0.0, 0.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# TrackingLoop injectable avoidance
# ---------------------------------------------------------------------------


class _ConstantAvoidance:
    """Stub AvoidanceStrategy returning a fixed turn-rate bias."""

    def __init__(self, bias: float) -> None:
        self.bias = bias
        self.call_count = 0

    def __call__(self, x: float, y: float, theta: float) -> float:
        self.call_count += 1
        return self.bias


def test_tracking_loop_uses_injected_avoidance() -> None:
    """Explicit avoidance replaces the default APF entirely."""
    stub = _ConstantAvoidance(bias=0.42)
    occ = _NearObstacleOccupancy(0.5, 0.0, clearance=2.0)
    loop = TrackingLoop(
        _make_vehicle(),
        PurePursuitController(lookahead_distance=2.0),
        cruise_speed=1.0,
        occupancy=occ,
        repulsion_gain=9.0,
        avoidance=stub,
    )
    assert loop._avoidance is stub
    assert loop._repulsion_turn_rate(0.0, 0.0, 0.0) == 0.42


def test_tracking_loop_step_applies_injected_avoidance() -> None:
    """step() blends the injected avoidance bias into turn_rate metrics."""
    stub = _ConstantAvoidance(bias=0.25)
    loop = TrackingLoop(
        _make_vehicle(),
        PurePursuitController(lookahead_distance=2.0),
        cruise_speed=1.0,
        avoidance=stub,
    )
    metrics = loop.step(STRAIGHT_PATH, dt=0.1)
    assert metrics["repulsion_turn_rate"] == 0.25
    assert stub.call_count == 1


def test_tracking_loop_noop_apf_disables_despite_occupancy_kwargs() -> None:
    """Injected no-op APF disables repulsion even when kwargs would enable it."""
    noop = ArtificialPotentialField(occupancy=None, repulsion_gain=0.0)
    loop = TrackingLoop(
        _make_vehicle(),
        PurePursuitController(lookahead_distance=2.0),
        cruise_speed=1.0,
        occupancy=_NearObstacleOccupancy(0.5, 0.0, clearance=2.0),
        repulsion_gain=2.0,
        avoidance=noop,
    )
    assert loop._repulsion_turn_rate(0.0, 0.0, 0.0) == 0.0
    metrics = loop.step(STRAIGHT_PATH, dt=0.1)
    assert metrics["repulsion_turn_rate"] == 0.0


def test_default_apf_matches_standalone_apf() -> None:
    """TrackingLoop default APF matches a standalone ArtificialPotentialField."""
    occ = _NearObstacleOccupancy(1.0, -1.0, clearance=4.0)
    gain = 1.75
    standalone = ArtificialPotentialField(occ, repulsion_gain=gain)
    loop = TrackingLoop(
        _make_vehicle(),
        PurePursuitController(lookahead_distance=2.0),
        cruise_speed=1.0,
        occupancy=occ,
        repulsion_gain=gain,
    )
    assert loop._repulsion_turn_rate(0.0, 0.0, 0.0) == standalone(
        0.0, 0.0, 0.0
    )
