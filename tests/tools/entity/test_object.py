"""Tests for arco.tools.entity.object — passive Object entity."""

from __future__ import annotations

import pytest

from arco.simulator.entity import BoxGeometry, Object, SphereGeometry


def _obj(**kwargs) -> Object:
    defaults = dict(
        name="crate",
        geometry=BoxGeometry(half_extents=(0.5, 0.5)),
        state=[1.0, 2.0, 0.0],
        mass=5.0,
    )
    defaults.update(kwargs)
    return Object(**defaults)


def test_object_stores_state() -> None:
    """State is preserved after construction."""
    obj = _obj(state=[3.0, 4.0, 1.57])
    assert obj.state == pytest.approx([3.0, 4.0, 1.57])


def test_object_mass_default() -> None:
    """Default mass is 1.0 kg."""
    obj = Object(
        name="ball",
        geometry=SphereGeometry(radius=0.1),
    )
    assert obj.mass == pytest.approx(1.0)


def test_object_to_dict_type() -> None:
    """to_dict includes type='Object'."""
    d = _obj().to_dict()
    assert d["type"] == "Object"


def test_object_to_dict_fields() -> None:
    """to_dict contains all expected fields."""
    obj = _obj()
    d = obj.to_dict()
    assert "name" in d
    assert "geometry" in d
    assert "state" in d
    assert "mass" in d


def test_object_round_trip_box() -> None:
    """from_dict(to_dict(obj)) reproduces BoxGeometry object."""
    obj = _obj(state=[1.0, 2.0, 0.5], mass=3.0)
    obj2 = Object.from_dict(obj.to_dict())
    assert obj2.name == obj.name
    assert obj2.state == pytest.approx(obj.state)
    assert obj2.mass == pytest.approx(obj.mass)
    assert isinstance(obj2.geometry, BoxGeometry)
    assert obj2.geometry.half_extents == pytest.approx(
        obj.geometry.half_extents
    )


def test_object_round_trip_sphere() -> None:
    """from_dict(to_dict(obj)) reproduces SphereGeometry object."""
    obj = Object(
        name="ball",
        geometry=SphereGeometry(radius=0.2),
        state=[0.0, 0.0, 0.0],
        mass=0.5,
    )
    obj2 = Object.from_dict(obj.to_dict())
    assert isinstance(obj2.geometry, SphereGeometry)
    assert obj2.geometry.radius == pytest.approx(0.2)


def test_object_mutable_state() -> None:
    """State list can be mutated in-place."""
    obj = _obj(state=[0.0, 0.0, 0.0])
    obj.state[0] = 5.0
    assert obj.state[0] == pytest.approx(5.0)
