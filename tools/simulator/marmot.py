"""Marmot mascot drawing utilities for the ARCO simulator.

Provides a reusable function to render a flat marmot silhouette on a
transparent :class:`pygame.Surface`.  The fill colour is parameterised at
render time so the mascot can match any agent colour.

The marmot is drawn entirely with ``pygame.draw`` primitives — no gradients,
no shadows, flat fill — and is suitable for embedding inside a
:func:`renderer_gl.blit_overlay` call alongside any existing HUD element.
"""

from __future__ import annotations

import pygame


def draw_marmot_surface(
    size: int,
    color: tuple[int, int, int],
) -> pygame.Surface:
    """Return a transparent surface containing a flat marmot mascot.

    The marmot is drawn at *size* × *size* pixels using simple
    :mod:`pygame.draw` primitives (ellipses, circles).  The main body fill
    uses *color*; ears and paws use a slightly darker shade derived from
    *color*; eye and nose details use near-black.

    Args:
        size: Edge length of the returned square surface in pixels.
        color: RGB fill colour applied to the marmot body and head.

    Returns:
        SRCALPHA square surface with the marmot centred on a transparent
        background.
    """
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))

    s = size / 100.0  # pixels-per-unit scale factor

    def _px(v: float) -> int:
        return max(1, int(round(v * s)))

    r, g, b = color
    dark: tuple[int, int, int] = (
        max(0, r - 55),
        max(0, g - 55),
        max(0, b - 55),
    )
    light: tuple[int, int, int] = (
        min(255, r + 50),
        min(255, g + 50),
        min(255, b + 50),
    )

    # --- Body (wide ellipse) ------------------------------------------------
    pygame.draw.ellipse(
        surf,
        color,
        pygame.Rect(_px(21), _px(50), _px(58), _px(43)),
    )

    # --- Head (circle) ------------------------------------------------------
    pygame.draw.circle(surf, color, (_px(50), _px(33)), _px(21))

    # --- Ears ---------------------------------------------------------------
    pygame.draw.circle(surf, dark, (_px(34), _px(15)), _px(9))
    pygame.draw.circle(surf, dark, (_px(66), _px(15)), _px(9))
    # Inner ear highlight
    pygame.draw.circle(surf, light, (_px(34), _px(15)), _px(5))
    pygame.draw.circle(surf, light, (_px(66), _px(15)), _px(5))

    # --- Front paws / arms --------------------------------------------------
    pygame.draw.ellipse(
        surf,
        dark,
        pygame.Rect(_px(13), _px(63), _px(17), _px(12)),
    )
    pygame.draw.ellipse(
        surf,
        dark,
        pygame.Rect(_px(70), _px(63), _px(17), _px(12)),
    )

    # --- Hind feet ----------------------------------------------------------
    pygame.draw.ellipse(
        surf,
        dark,
        pygame.Rect(_px(24), _px(88), _px(20), _px(9)),
    )
    pygame.draw.ellipse(
        surf,
        dark,
        pygame.Rect(_px(56), _px(88), _px(20), _px(9)),
    )

    # --- Face details (near-black) ------------------------------------------
    _eye = (20, 20, 20)
    _nose = (30, 15, 15)
    pygame.draw.circle(surf, _eye, (_px(43), _px(29)), max(1, _px(3)))
    pygame.draw.circle(surf, _eye, (_px(57), _px(29)), max(1, _px(3)))
    pygame.draw.circle(surf, _nose, (_px(50), _px(37)), max(1, _px(2)))

    return surf
