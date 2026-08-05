"""Tests for the GTA2-style car sprite surface generation."""

from __future__ import annotations

import pytest

pygame = pytest.importorskip("pygame")
pytest.importorskip("OpenGL")

from arco.simulator.sim.car_sprite import (  # noqa: E402
    _GRID_L,
    _GRID_W,
    make_car_sprite_surface,
)

_BLUE = (59, 130, 246)
_GREEN = (34, 197, 94)


def _logical(surf: pygame.Surface, x: int, y: int, scale: int = 4):
    """Sample the center of logical pixel (x, y) of an upscaled sprite."""
    return surf.get_at((x * scale + scale // 2, y * scale + scale // 2))


def test_sprite_surface_size_and_alpha() -> None:
    surf = make_car_sprite_surface(_BLUE)
    assert surf.get_size() == (_GRID_L * 4, _GRID_W * 4)
    # Sprite corners are transparent (rounded body, no full-rect blob).
    assert _logical(surf, 0, 0).a == 0
    assert _logical(surf, _GRID_L - 1, _GRID_W - 1).a == 0


def test_sprite_has_tires_glass_and_lights() -> None:
    surf = make_car_sprite_surface(_BLUE)
    # Tires: near-black pixels poking out on the sides.
    tire = _logical(surf, 4, 0)
    assert tire.a == 255
    assert max(tire.r, tire.g, tire.b) < 60
    # Windshield glass is a desaturated dark blue-gray, not body blue.
    glass = _logical(surf, 14, 5)
    assert glass.b > glass.r
    assert max(glass.r, glass.g, glass.b) < 160
    # Headlights (front, +X) are pale yellow; taillights are red.
    head = _logical(surf, 22, 3)
    assert head.r > 200 and head.g > 200
    tail = _logical(surf, 1, 3)
    assert tail.r > 150 and tail.g < 90


def test_sprite_body_uses_racer_color() -> None:
    blue = make_car_sprite_surface(_BLUE)
    green = make_car_sprite_surface(_GREEN)
    body_blue = _logical(blue, 18, 5)
    body_green = _logical(green, 18, 5)
    # Hue follows the racer color (blue-dominant vs green-dominant).
    assert body_blue.b > body_blue.g > body_blue.r
    assert body_green.g > body_green.r
    assert body_green.g > body_green.b
    # Distinct racer colors must yield distinct sprites.
    assert (body_blue.r, body_blue.g, body_blue.b) != (
        body_green.r,
        body_green.g,
        body_green.b,
    )


def test_sprite_roof_is_lighter_than_sides() -> None:
    surf = make_car_sprite_surface(_BLUE)
    roof = _logical(surf, 10, 5)
    side = _logical(surf, 10, 2)
    assert (roof.r + roof.g + roof.b) > (side.r + side.g + side.b)


def test_sprite_custom_scale() -> None:
    surf = make_car_sprite_surface(_BLUE, scale=2)
    assert surf.get_size() == (_GRID_L * 2, _GRID_W * 2)
