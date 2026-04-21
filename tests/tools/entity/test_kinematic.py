"""Tests for arco.tools.entity.kinematic — joints, links, end-effectors."""

from __future__ import annotations

import math

import pytest

from arco.simulator.entity import (
    BoxGeometry,
    EndEffector,
    KinematicChain,
    Link,
    PrismaticJoint,
    RevoluteJoint,
    SphereGeometry,
)
from arco.simulator.entity.kinematic import _joint_from_dict

# ---------------------------------------------------------------------------
# RevoluteJoint
# ---------------------------------------------------------------------------


def _revolute(**kwargs) -> RevoluteJoint:
    defaults = dict(
        name="j1",
        geometry=SphereGeometry(radius=0.05),
        state=[0.0],
        min_position=-math.pi,
        max_position=math.pi,
        parent_link="base",
        child_link="link1",
    )
    defaults.update(kwargs)
    return RevoluteJoint(**defaults)


def test_revolute_position_getter() -> None:
    """position property returns first state element."""
    j = _revolute(state=[1.2])
    assert j.position == pytest.approx(1.2)


def test_revolute_position_setter_clamps() -> None:
    """position setter clamps to [min, max]."""
    j = _revolute(min_position=0.0, max_position=1.0)
    j.position = 5.0
    assert j.position == pytest.approx(1.0)
    j.position = -5.0
    assert j.position == pytest.approx(0.0)


def test_revolute_to_dict_type() -> None:
    """to_dict includes type='RevoluteJoint'."""
    d = _revolute().to_dict()
    assert d["type"] == "RevoluteJoint"


def test_revolute_round_trip() -> None:
    """from_dict(to_dict(j)) reproduces all fields."""
    j = _revolute(state=[0.5])
    d = j.to_dict()
    j2 = RevoluteJoint.from_dict(d)
    assert j2.name == j.name
    assert j2.position == pytest.approx(j.position)
    assert j2.min_position == pytest.approx(j.min_position)
    assert j2.max_position == pytest.approx(j.max_position)
    assert j2.parent_link == j.parent_link
    assert j2.child_link == j.child_link


# ---------------------------------------------------------------------------
# PrismaticJoint
# ---------------------------------------------------------------------------


def _prismatic(**kwargs) -> PrismaticJoint:
    defaults = dict(
        name="p1",
        geometry=BoxGeometry(half_extents=(0.02, 0.02)),
        state=[0.0],
        min_position=0.0,
        max_position=1.0,
        parent_link="base",
        child_link="slider",
    )
    defaults.update(kwargs)
    return PrismaticJoint(**defaults)


def test_prismatic_position_getter() -> None:
    """position property returns first state element."""
    j = _prismatic(state=[0.3])
    assert j.position == pytest.approx(0.3)


def test_prismatic_position_setter_clamps() -> None:
    """position setter clamps to [min, max]."""
    j = _prismatic(min_position=0.0, max_position=0.5)
    j.position = 1.0
    assert j.position == pytest.approx(0.5)


def test_prismatic_to_dict_type() -> None:
    """to_dict includes type='PrismaticJoint'."""
    d = _prismatic().to_dict()
    assert d["type"] == "PrismaticJoint"


def test_prismatic_round_trip() -> None:
    """from_dict(to_dict(j)) reproduces all fields."""
    j = _prismatic(state=[0.2])
    d = j.to_dict()
    j2 = PrismaticJoint.from_dict(d)
    assert j2.name == j.name
    assert j2.position == pytest.approx(j.position)


# ---------------------------------------------------------------------------
# _joint_from_dict dispatch
# ---------------------------------------------------------------------------


def test_joint_from_dict_revolute() -> None:
    """Dispatch returns RevoluteJoint for correct type."""
    d = {
        "type": "RevoluteJoint",
        "name": "j",
        "geometry": {"type": "sphere", "radius": 0.01},
        "state": [0.0],
        "min_position": -3.14,
        "max_position": 3.14,
        "parent_link": "a",
        "child_link": "b",
    }
    j = _joint_from_dict(d)
    assert isinstance(j, RevoluteJoint)


def test_joint_from_dict_prismatic() -> None:
    """Dispatch returns PrismaticJoint for correct type."""
    d = {
        "type": "PrismaticJoint",
        "name": "p",
        "geometry": {"type": "box", "half_extents": [0.01, 0.01]},
        "state": [0.0],
        "min_position": 0.0,
        "max_position": 1.0,
        "parent_link": "a",
        "child_link": "b",
    }
    j = _joint_from_dict(d)
    assert isinstance(j, PrismaticJoint)


def test_joint_from_dict_unknown_raises() -> None:
    """Dispatch raises ValueError for unknown joint type."""
    with pytest.raises(ValueError, match="Unknown joint type"):
        _joint_from_dict({"type": "BallJoint"})


# ---------------------------------------------------------------------------
# Link
# ---------------------------------------------------------------------------


def test_link_to_dict_type() -> None:
    """to_dict includes type='Link'."""
    lk = Link(
        name="lk1",
        geometry=BoxGeometry(half_extents=(0.1, 0.5)),
        state=[0.0, 0.0, 0.0],
        mass=1.0,
        parent_joint="j1",
    )
    d = lk.to_dict()
    assert d["type"] == "Link"
    assert d["parent_joint"] == "j1"


def test_link_round_trip() -> None:
    """from_dict(to_dict(link)) reproduces all fields."""
    lk = Link(
        name="lk1",
        geometry=BoxGeometry(half_extents=(0.2, 0.4)),
        state=[1.0, 2.0, 0.5],
        mass=2.5,
        parent_joint="j0",
    )
    lk2 = Link.from_dict(lk.to_dict())
    assert lk2.name == lk.name
    assert lk2.state == pytest.approx(lk.state)
    assert lk2.mass == pytest.approx(lk.mass)
    assert lk2.parent_joint == lk.parent_joint


# ---------------------------------------------------------------------------
# EndEffector
# ---------------------------------------------------------------------------


def test_end_effector_to_dict_type() -> None:
    """to_dict includes type='EndEffector'."""
    ee = EndEffector(
        name="tool",
        geometry=SphereGeometry(radius=0.02),
        state=[1.0, 2.0, 0.0],
        parent_joint="j3",
    )
    d = ee.to_dict()
    assert d["type"] == "EndEffector"


def test_end_effector_round_trip() -> None:
    """from_dict(to_dict(ee)) reproduces all fields."""
    ee = EndEffector(
        name="tip",
        geometry=SphereGeometry(radius=0.01),
        state=[0.5, 0.5, 0.0],
        parent_joint="j2",
    )
    ee2 = EndEffector.from_dict(ee.to_dict())
    assert ee2.name == ee.name
    assert ee2.state == pytest.approx(ee.state)
    assert ee2.parent_joint == ee.parent_joint


# ---------------------------------------------------------------------------
# KinematicChain
# ---------------------------------------------------------------------------


def _make_2dof_revolute_arm() -> KinematicChain:
    """Build a minimal 2-DoF revolute chain for testing."""
    links = [
        Link(
            name="base",
            geometry=BoxGeometry(half_extents=(0.1, 0.1)),
            state=[0.0, 0.0, 0.0],
        ),
        Link(
            name="forearm",
            geometry=BoxGeometry(half_extents=(0.5, 0.05)),
            state=[0.0, 0.0, 0.0],
            parent_joint="shoulder",
        ),
    ]
    joints = [
        RevoluteJoint(
            name="shoulder",
            geometry=SphereGeometry(radius=0.05),
            state=[0.0],
            min_position=-math.pi,
            max_position=math.pi,
            parent_link="base",
            child_link="forearm",
        ),
        RevoluteJoint(
            name="elbow",
            geometry=SphereGeometry(radius=0.04),
            state=[0.0],
            min_position=-math.pi / 2,
            max_position=math.pi / 2,
            parent_link="forearm",
            child_link="hand",
        ),
    ]
    ee = EndEffector(
        name="hand",
        geometry=SphereGeometry(radius=0.03),
        state=[0.0, 0.0, 0.0],
        parent_joint="elbow",
    )
    return KinematicChain(
        name="arm", links=links, joints=joints, end_effector=ee
    )


def test_kinematic_chain_dof() -> None:
    """dof returns the number of joints."""
    chain = _make_2dof_revolute_arm()
    assert chain.dof == 2


def test_kinematic_chain_joint_positions() -> None:
    """joint_positions returns correct initial positions."""
    chain = _make_2dof_revolute_arm()
    assert chain.joint_positions == pytest.approx([0.0, 0.0])


def test_kinematic_chain_joint_positions_after_set() -> None:
    """joint_positions reflects updated joint states."""
    chain = _make_2dof_revolute_arm()
    chain.joints[0].position = 0.5
    chain.joints[1].position = -0.3
    assert chain.joint_positions == pytest.approx([0.5, -0.3])


def test_kinematic_chain_round_trip() -> None:
    """from_dict(to_dict(chain)) reproduces all components."""
    chain = _make_2dof_revolute_arm()
    chain.joints[0].position = 0.7
    d = chain.to_dict()
    chain2 = KinematicChain.from_dict(d)
    assert chain2.name == chain.name
    assert chain2.dof == chain.dof
    assert chain2.joint_positions == pytest.approx(chain.joint_positions)
    assert chain2.end_effector is not None
    assert chain2.end_effector.name == "hand"


def test_kinematic_chain_no_end_effector() -> None:
    """Chain without end-effector serialises and deserialises correctly."""
    chain = KinematicChain(
        name="bare",
        links=[
            Link(
                name="l1",
                geometry=BoxGeometry(half_extents=(0.1,)),
            )
        ],
        joints=[],
        end_effector=None,
    )
    d = chain.to_dict()
    chain2 = KinematicChain.from_dict(d)
    assert chain2.end_effector is None
