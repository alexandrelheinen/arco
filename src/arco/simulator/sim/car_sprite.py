"""Procedural GTA2-style top-down car sprites for race vehicles.

Generates small pixel-art car sprites (body colored per racer, dark
glass, black tires, head/taillights) as pygame surfaces, uploads them as
OpenGL textures on first use, and draws them as oriented textured quads
in world coordinates.

Surface generation is pure pygame (no GL context required) so it stays
unit-testable on headless CI runners; only :func:`draw_car_sprite`
touches OpenGL state.
"""

from __future__ import annotations

import math

import pygame
from OpenGL.GL import (  # type: ignore[import-untyped]
    GL_MODULATE,
    GL_NEAREST,
    GL_QUADS,
    GL_RGBA,
    GL_TEXTURE_2D,
    GL_TEXTURE_ENV,
    GL_TEXTURE_ENV_MODE,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_UNSIGNED_BYTE,
    glBegin,
    glBindTexture,
    glColor4f,
    glDisable,
    glEnable,
    glEnd,
    glGenTextures,
    glTexCoord2f,
    glTexEnvf,
    glTexImage2D,
    glTexParameteri,
    glVertex2f,
)

# Logical pixel grid of the sprite: car points +X (heading = 0 = east).
_GRID_L = 24
_GRID_W = 12
# Upscale factor from the logical grid to the generated surface.
_SPRITE_SCALE = 4

# Fixed palette parts (RGB).
_C_TIRE = (42, 44, 50)
_C_GLASS = (52, 74, 94)
_C_GLASS_HI = (86, 116, 140)
_C_HEADLIGHT = (255, 244, 176)
_C_TAILLIGHT = (208, 48, 40)
_C_OUTLINE = (16, 16, 20)

_texture_cache: dict[tuple[int, int, int], int] = {}


def _mix(
    a: tuple[int, int, int], b: tuple[int, int, int], t: float
) -> tuple[int, int, int]:
    """Linearly interpolate two RGB colors.

    Args:
        a: Start color.
        b: End color.
        t: Blend factor in ``[0, 1]`` (0 → *a*, 1 → *b*).

    Returns:
        Blended RGB tuple.
    """
    t = float(min(max(t, 0.0), 1.0))
    return (
        int(round(a[0] + (b[0] - a[0]) * t)),
        int(round(a[1] + (b[1] - a[1]) * t)),
        int(round(a[2] + (b[2] - a[2]) * t)),
    )


def make_car_sprite_surface(
    color: tuple[int, int, int],
    *,
    scale: int = _SPRITE_SCALE,
) -> pygame.Surface:
    """Build a GTA2-style top-down car sprite pointing east (+X).

    The sprite is drawn on a ``24 × 12`` logical pixel grid (length ×
    width) and upscaled with nearest-neighbor so the pixel-art look
    survives: black outline and tires, per-racer body color with hood /
    roof shading, dark windshield and rear glass, pale headlights, and
    red taillights.

    Args:
        color: Base body RGB color in ``[0, 255]``.
        scale: Integer upscale factor from the logical grid.

    Returns:
        ``pygame.Surface`` with per-pixel alpha, size
        ``(24 · scale, 12 · scale)``.
    """
    body = _mix(color, (255, 255, 255), 0.20)
    body_dark = _mix(color, (0, 0, 0), 0.35)
    roof = _mix(color, (255, 255, 255), 0.42)
    roof_edge = _mix(color, (255, 255, 255), 0.05)

    surf = pygame.Surface((_GRID_L, _GRID_W), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))

    def px(x: int, y: int, c: tuple[int, int, int]) -> None:
        surf.set_at((x, y), (*c, 255))

    def rect(x0: int, y0: int, x1: int, y1: int, c) -> None:
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                px(x, y, c)

    # Body outline (rows 1..10), rounded by trimming the four corners.
    rect(0, 1, 23, 10, _C_OUTLINE)

    # Body fill inside the outline.
    rect(1, 2, 22, 9, body)

    # Rounded nose and tail: trim outline corners to transparent and pull
    # the fill corners back to outline so the silhouette curves.
    for x, y in ((0, 1), (0, 10), (23, 1), (23, 10)):
        surf.set_at((x, y), (0, 0, 0, 0))
    for x, y in ((1, 2), (1, 9), (22, 2), (22, 9)):
        px(x, y, _C_OUTLINE)

    # Body side shading rows for a hint of volume.
    rect(2, 2, 21, 2, body_dark)
    rect(2, 9, 21, 9, body_dark)

    # Hood highlight (front deck) and a subtle trunk panel.
    rect(17, 4, 20, 7, roof_edge)
    rect(3, 4, 5, 7, roof_edge)

    # Tires — black blocks poking outside the body, drawn over it.
    rect(3, 0, 6, 1, _C_TIRE)
    rect(3, 10, 6, 11, _C_TIRE)
    rect(16, 0, 19, 1, _C_TIRE)
    rect(16, 10, 19, 11, _C_TIRE)

    # Cabin: dark glass inset with body-colored pillars left visible.
    rect(14, 3, 15, 8, _C_GLASS)  # windshield (front of cabin)
    rect(14, 4, 14, 7, _C_GLASS_HI)
    rect(7, 3, 8, 8, _C_GLASS)  # rear window
    rect(9, 3, 12, 3, _C_GLASS)  # side windows
    rect(9, 8, 12, 8, _C_GLASS)

    # Cabin roof between the glass panes, lighter center.
    rect(9, 4, 12, 7, roof)

    # Headlights (front = +X) and taillights (rear).
    rect(22, 3, 22, 4, _C_HEADLIGHT)
    rect(22, 7, 22, 8, _C_HEADLIGHT)
    rect(1, 3, 1, 4, _C_TAILLIGHT)
    rect(1, 7, 1, 8, _C_TAILLIGHT)

    scale = max(int(scale), 1)
    return pygame.transform.scale(surf, (_GRID_L * scale, _GRID_W * scale))


def _get_car_texture(color: tuple[int, int, int]) -> int:
    """Return (building if needed) the GL texture for a body color.

    Args:
        color: Base body RGB color in ``[0, 255]``.

    Returns:
        OpenGL texture id.
    """
    key = (int(color[0]), int(color[1]), int(color[2]))
    tex_id = _texture_cache.get(key)
    if tex_id is not None:
        return tex_id
    surf = make_car_sprite_surface(key)
    w, h = surf.get_size()
    data = pygame.image.tostring(surf, "RGBA", True)
    tex_id = int(glGenTextures(1))
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data
    )
    glBindTexture(GL_TEXTURE_2D, 0)
    _texture_cache[key] = tex_id
    return tex_id


def reset_texture_cache() -> None:
    """Forget cached texture ids (call after the GL context is torn down)."""
    _texture_cache.clear()


def draw_car_sprite(
    x: float,
    y: float,
    heading: float,
    half_l: float,
    half_w: float,
    color: tuple[int, int, int],
) -> None:
    """Draw a car sprite as an oriented textured quad in world meters.

    Args:
        x: Vehicle center x in world meters.
        y: Vehicle center y in world meters.
        heading: Vehicle heading in radians (0 = east).
        half_l: Half-length (forward) in world meters.
        half_w: Half-width (lateral) in world meters.
        color: Base body RGB color in ``[0, 255]``.
    """
    tex_id = _get_car_texture(color)
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    # Local corners (front-left, front-right, rear-right, rear-left) and
    # matching texture coordinates: +X in sprite space is the car front.
    corners = (
        (half_l, half_w, 1.0, 1.0),
        (half_l, -half_w, 1.0, 0.0),
        (-half_l, -half_w, 0.0, 0.0),
        (-half_l, half_w, 0.0, 1.0),
    )
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
    glColor4f(1.0, 1.0, 1.0, 1.0)
    glBegin(GL_QUADS)
    for lx, ly, u, v in corners:
        glTexCoord2f(u, v)
        glVertex2f(x + lx * cos_h - ly * sin_h, y + lx * sin_h + ly * cos_h)
    glEnd()
    glBindTexture(GL_TEXTURE_2D, 0)
    glDisable(GL_TEXTURE_2D)
