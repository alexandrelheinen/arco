"""Tests for arco.tools.entity.agent — DubinsAgent and CartesianAgent."""

from __future__ import annotations

import math

import pytest

from arco.simulator.entity import (
    BoxGeometry,
    CartesianAgent,
    DubinsAgent,
    SphereGeometry,
)

# ---------------------------------------------------------------------------
# DubinsAgent
# ---------------------------------------------------------------------------


def _dubins(
    state: list[float] | None = None,
    max_speed: float = 5.0,
    max_turn_rate: float = math.pi,
) -> DubinsAgent:
    return DubinsAgent(
        name="veh",
        geometry=BoxGeometry(half_extents=(1.0, 0.5)),
        state=state or [0.0, 0.0, 0.0],
        max_speed=max_speed,
        max_turn_rate=max_turn_rate,
    )


def test_dubins_initial_state() -> None:
    """Initial state is stored correctly."""
    agent = _dubins(state=[1.0, 2.0, 0.5])
    assert agent.state == pytest.approx([1.0, 2.0, 0.5])


def test_dubins_step_straight() -> None:
    """No turn-rate → vehicle moves straight along heading."""
    agent = _dubins(state=[0.0, 0.0, 0.0])
    agent.speed = 1.0
    agent.step([0.0, 0.0], dt=1.0)  # constant speed, no accel
    assert agent.state[0] == pytest.approx(1.0, abs=1e-9)
    assert agent.state[1] == pytest.approx(0.0, abs=1e-9)


def test_dubins_step_accel() -> None:
    """Positive acceleration increases speed."""
    agent = _dubins()
    agent.step([0.0, 2.0], dt=0.5)
    assert agent.speed == pytest.approx(1.0, abs=1e-9)


def test_dubins_speed_clamp() -> None:
    """Speed is clamped to max_speed."""
    agent = _dubins(max_speed=2.0)
    agent.step([0.0, 100.0], dt=1.0)
    assert agent.speed == pytest.approx(2.0)


def test_dubins_turn_rate_clamp() -> None:
    """Turn-rate is clamped to ±max_turn_rate."""
    agent = _dubins(state=[0.0, 0.0, 0.0], max_turn_rate=0.1)
    agent.step([999.0, 0.0], dt=1.0)
    assert agent.heading == pytest.approx(0.1)


def test_dubins_heading_update() -> None:
    """Heading changes by turn_rate * dt."""
    agent = _dubins(state=[0.0, 0.0, 0.0])
    agent.step([1.0, 0.0], dt=0.5)
    assert agent.heading == pytest.approx(0.5)


def test_dubins_named_properties() -> None:
    """x, y, heading properties alias state[0], state[1], state[2]."""
    agent = _dubins(state=[3.0, 4.0, 1.5])
    assert agent.x == pytest.approx(3.0)
    assert agent.y == pytest.approx(4.0)
    assert agent.heading == pytest.approx(1.5)
    agent.x = 10.0
    agent.y = 20.0
    agent.heading = 0.0
    assert agent.state == pytest.approx([10.0, 20.0, 0.0])


def test_dubins_step_bad_state_raises() -> None:
    """step raises ValueError when state has fewer than 3 elements."""
    agent = DubinsAgent(
        name="bad",
        geometry=BoxGeometry(half_extents=(1.0, 0.5)),
        state=[0.0, 0.0],
    )
    with pytest.raises(ValueError, match="requires state"):
        agent.step([0.0, 0.0], dt=0.1)


def test_dubins_step_bad_control_raises() -> None:
    """step raises ValueError when control has fewer than 2 elements."""
    agent = _dubins()
    with pytest.raises(ValueError, match="control must be"):
        agent.step([0.0], dt=0.1)


def test_dubins_to_dict_type() -> None:
    """to_dict includes type='DubinsAgent'."""
    d = _dubins().to_dict()
    assert d["type"] == "DubinsAgent"


def test_dubins_round_trip() -> None:
    """from_dict(to_dict(agent)) reproduces all fields."""
    agent = _dubins(state=[1.0, 2.0, 0.3], max_speed=3.0, max_turn_rate=1.0)
    agent.speed = 1.5
    d = agent.to_dict()
    agent2 = DubinsAgent.from_dict(d)
    assert agent2.name == agent.name
    assert agent2.state == pytest.approx(agent.state)
    assert agent2.max_speed == pytest.approx(agent.max_speed)
    assert agent2.max_turn_rate == pytest.approx(agent.max_turn_rate)
    assert agent2.speed == pytest.approx(agent.speed)
    assert isinstance(agent2.geometry, BoxGeometry)


def test_dubins_sphere_geometry() -> None:
    """DubinsAgent also accepts SphereGeometry."""
    agent = DubinsAgent(
        name="ball",
        geometry=SphereGeometry(radius=0.3),
        state=[0.0, 0.0, 0.0],
    )
    d = agent.to_dict()
    agent2 = DubinsAgent.from_dict(d)
    assert isinstance(agent2.geometry, SphereGeometry)
    assert agent2.geometry.radius == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# CartesianAgent
# ---------------------------------------------------------------------------


def _cartesian(
    state: list[float] | None = None,
    max_speed: float = 1.0,
    bandwidth: float = 10.0,
) -> CartesianAgent:
    return CartesianAgent(
        name="robot",
        geometry=BoxGeometry(half_extents=(0.1, 0.1, 0.1)),
        state=state or [0.0, 0.0, 0.0],
        max_speed=max_speed,
        bandwidth=bandwidth,
    )


def test_cartesian_initial_state() -> None:
    """Initial state is stored correctly."""
    agent = _cartesian(state=[1.0, 2.0, 3.0])
    assert agent.state == pytest.approx([1.0, 2.0, 3.0])


def test_cartesian_velocities_initialised() -> None:
    """Velocity vector is initialised to zero with matching length."""
    agent = _cartesian(state=[0.0, 0.0, 0.0])
    assert len(agent.velocities) == 3
    assert all(v == 0.0 for v in agent.velocities)


def test_cartesian_step_integrates_position() -> None:
    """After sufficient time with command speed, position increases."""
    agent = _cartesian(state=[0.0], max_speed=2.0, bandwidth=100.0)
    for _ in range(100):
        agent.step([1.0], dt=0.01)
    assert agent.state[0] > 0.5  # converged towards 1 m/s * 1 s


def test_cartesian_speed_clamp() -> None:
    """Velocity is clamped to ±max_speed."""
    agent = _cartesian(state=[0.0], max_speed=0.5, bandwidth=1000.0)
    agent.step([999.0], dt=1.0)
    assert abs(agent.velocities[0]) <= 0.5 + 1e-9


def test_cartesian_step_dimension_mismatch_raises() -> None:
    """step raises ValueError when control length != state length."""
    agent = _cartesian(state=[0.0, 0.0])
    with pytest.raises(ValueError, match="control has"):
        agent.step([0.0], dt=0.1)


def test_cartesian_to_dict_type() -> None:
    """to_dict includes type='CartesianAgent'."""
    d = _cartesian().to_dict()
    assert d["type"] == "CartesianAgent"


def test_cartesian_round_trip() -> None:
    """from_dict(to_dict(agent)) reproduces all fields."""
    agent = _cartesian(state=[1.0, 2.0], max_speed=0.5, bandwidth=5.0)
    agent.step([0.3, 0.3], dt=0.1)
    d = agent.to_dict()
    agent2 = CartesianAgent.from_dict(d)
    assert agent2.name == agent.name
    assert agent2.state == pytest.approx(agent.state)
    assert agent2.max_speed == pytest.approx(agent.max_speed)
    assert agent2.bandwidth == pytest.approx(agent.bandwidth)
    assert agent2.velocities == pytest.approx(agent.velocities)
