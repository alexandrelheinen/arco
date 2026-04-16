"""Tests for arco.tools.entity.base — geometry descriptors and Entity ABC."""

from __future__ import annotations

import pytest

from arco.tools.entity import (
    BoxGeometry,
    SphereGeometry,
    geometry_from_dict,
)

# ---------------------------------------------------------------------------
# BoxGeometry
# ---------------------------------------------------------------------------


def test_box_geometry_stores_half_extents() -> None:
    """Half-extents are preserved as a tuple."""
    g = BoxGeometry(half_extents=(1.0, 2.0))
    assert g.half_extents == (1.0, 2.0)


def test_box_geometry_3d() -> None:
    """Three-dimensional half-extents are supported."""
    g = BoxGeometry(half_extents=(0.5, 0.5, 0.5))
    assert len(g.half_extents) == 3


def test_box_geometry_to_dict() -> None:
    """to_dict returns correct type and half_extents."""
    g = BoxGeometry(half_extents=(1.0, 2.0))
    d = g.to_dict()
    assert d["type"] == "box"
    assert d["half_extents"] == [1.0, 2.0]


def test_box_geometry_round_trip() -> None:
    """from_dict(to_dict(g)) == g."""
    g = BoxGeometry(half_extents=(3.0, 4.0))
    g2 = BoxGeometry.from_dict(g.to_dict())
    assert g2.half_extents == pytest.approx(g.half_extents)


def test_box_geometry_from_dict_raises_on_missing_key() -> None:
    """from_dict raises KeyError when half_extents is absent."""
    with pytest.raises(KeyError):
        BoxGeometry.from_dict({"type": "box"})


# ---------------------------------------------------------------------------
# SphereGeometry
# ---------------------------------------------------------------------------


def test_sphere_geometry_stores_radius() -> None:
    """Radius is preserved."""
    g = SphereGeometry(radius=2.5)
    assert g.radius == pytest.approx(2.5)


def test_sphere_geometry_to_dict() -> None:
    """to_dict returns correct type and radius."""
    g = SphereGeometry(radius=1.0)
    d = g.to_dict()
    assert d["type"] == "sphere"
    assert d["radius"] == pytest.approx(1.0)


def test_sphere_geometry_round_trip() -> None:
    """from_dict(to_dict(g)) == g."""
    g = SphereGeometry(radius=0.3)
    g2 = SphereGeometry.from_dict(g.to_dict())
    assert g2.radius == pytest.approx(g.radius)


def test_sphere_geometry_from_dict_raises_on_missing_key() -> None:
    """from_dict raises KeyError when radius is absent."""
    with pytest.raises(KeyError):
        SphereGeometry.from_dict({"type": "sphere"})


# ---------------------------------------------------------------------------
# geometry_from_dict dispatch
# ---------------------------------------------------------------------------


def test_geometry_from_dict_box() -> None:
    """geometry_from_dict dispatches to BoxGeometry."""
    d = {"type": "box", "half_extents": [1.0, 1.0]}
    g = geometry_from_dict(d)
    assert isinstance(g, BoxGeometry)


def test_geometry_from_dict_sphere() -> None:
    """geometry_from_dict dispatches to SphereGeometry."""
    d = {"type": "sphere", "radius": 0.5}
    g = geometry_from_dict(d)
    assert isinstance(g, SphereGeometry)


def test_geometry_from_dict_unknown_raises() -> None:
    """geometry_from_dict raises ValueError for unknown type."""
    with pytest.raises(ValueError, match="Unknown geometry type"):
        geometry_from_dict({"type": "cylinder"})


def test_geometry_from_dict_missing_type_raises() -> None:
    """geometry_from_dict raises KeyError when type key is absent."""
    with pytest.raises(KeyError):
        geometry_from_dict({"half_extents": [1.0]})
