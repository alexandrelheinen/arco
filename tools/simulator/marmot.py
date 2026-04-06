"""Marmot mascot renderer for the ARCO simulator.

Provides :func:`make_marmot_surface` — a factory that returns a flat,
minimal marmot silhouette as a pygame SRCALPHA surface.  The fill colour
is fully parameterised so the mascot can be tinted to match any winner's
colour at render time.

The design is intentionally simple (no gradients, no shadows, flat fill)
so it reads clearly at figure scale and stays visually consistent with the
rest of the simulator HUD.
"""

from __future__ import annotations

import pygame


def _lighten(
    color: tuple[int, int, int],
    factor: float = 1.35,
) -> tuple[int, int, int]:
    """Return a brightened copy of *color*.

    Args:
        color: Source RGB tuple (0–255 per channel).
        factor: Multiplier applied to each channel (>1 brightens).

    Returns:
        Clamped RGB tuple.
    """
    return (
        min(255, int(color[0] * factor)),
        min(255, int(color[1] * factor)),
        min(255, int(color[2] * factor)),
    )


def make_marmot_surface(
    color: tuple[int, int, int],
    height: int = 100,
) -> pygame.Surface:
    """Build a pygame SRCALPHA surface with a marmot mascot in *color*.

    The mascot is drawn flat (no gradients, no shadows) using
    :mod:`pygame.draw` primitives scaled from an 80 × 110 baseline
    design.  It is suitable for blitting directly into any overlay
    surface.

    Args:
        color: RGB fill colour for the marmot body (0–255 per channel).
        height: Desired surface height in pixels (width scales
            proportionally from the 80 × 110 baseline).

    Returns:
        Transparent SRCALPHA pygame surface containing the marmot.
    """
    W0, H0 = 80, 110  # baseline design dimensions
    scale = height / H0
    W = max(4, int(W0 * scale))
    H = max(4, height)

    surf = pygame.Surface((W, H), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))

    def s(v: float) -> int:
        return max(1, int(v * scale))

    cx = W // 2
    _DARK = (20, 20, 30)
    belly = _lighten(color, 1.4)
    shine = (230, 235, 255)

    # Ears (two small circles, placed behind the head)
    pygame.draw.circle(surf, color, (cx - s(13), s(18)), s(9))
    pygame.draw.circle(surf, color, (cx + s(13), s(18)), s(9))

    # Head
    pygame.draw.circle(surf, color, (cx, s(34)), s(20))

    # Body (tall ellipse)
    pygame.draw.ellipse(
        surf, color, pygame.Rect(cx - s(22), s(49), s(44), s(50))
    )

    # Arms (narrow ellipses on each side of the body)
    pygame.draw.ellipse(
        surf, color, pygame.Rect(cx - s(36), s(57), s(15), s(28))
    )
    pygame.draw.ellipse(
        surf, color, pygame.Rect(cx + s(21), s(57), s(15), s(28))
    )

    # Legs (rounded rectangles at the bottom)
    pygame.draw.rect(
        surf,
        color,
        pygame.Rect(cx - s(18), s(90), s(14), s(18)),
        border_radius=s(4),
    )
    pygame.draw.rect(
        surf,
        color,
        pygame.Rect(cx + s(4), s(90), s(14), s(18)),
        border_radius=s(4),
    )

    # Belly highlight (slightly brighter oval on the torso)
    pygame.draw.ellipse(
        surf, belly, pygame.Rect(cx - s(11), s(57), s(22), s(28))
    )

    # Eyes
    eye_r = s(4)
    pygame.draw.circle(surf, _DARK, (cx - s(7), s(31)), eye_r)
    pygame.draw.circle(surf, _DARK, (cx + s(7), s(31)), eye_r)
    shine_r = max(1, s(2))
    pygame.draw.circle(surf, shine, (cx - s(6), s(30)), shine_r)
    pygame.draw.circle(surf, shine, (cx + s(8), s(30)), shine_r)

    # Nose
    pygame.draw.ellipse(
        surf, _DARK, pygame.Rect(cx - s(4), s(39), s(8), s(5))
    )

    return surf
