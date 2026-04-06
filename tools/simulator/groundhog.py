"""Groundhog mascot renderer for the ARCO simulator.

Provides :func:`make_groundhog_surface` — a factory that returns a flat,
minimal groundhog silhouette as a pygame SRCALPHA surface.  The fill colour
is fully parameterised so the mascot can be tinted to match any winner's
colour at render time.

The design is intentionally simple (no gradients, no shadows, flat fill)
so it reads clearly at figure scale and stays visually consistent with the
rest of the simulator HUD.

Groundhog anatomy (flat cartoon style):

- Very wide, low-profile head with no visible neck
- Prominent cheek pouches that flare out to the sides
- Small rounded ears at the top corners of the head
- Open mouth with large buck teeth (incisors) and tongue
- Chunky squat body sitting beneath the head
- Short arms raised to the sides
"""

from __future__ import annotations

import pygame


def _lighten(
    color: tuple[int, int, int],
    factor: float = 1.35,
) -> tuple[int, int, int]:
    """Return a brightened copy of *color*.

    Args:
        color: Source RGB tuple (0-255 per channel).
        factor: Multiplier applied to each channel (>1 brightens).

    Returns:
        Clamped RGB tuple.
    """
    return (
        min(255, int(color[0] * factor)),
        min(255, int(color[1] * factor)),
        min(255, int(color[2] * factor)),
    )


def _darken(
    color: tuple[int, int, int],
    factor: float = 0.6,
) -> tuple[int, int, int]:
    """Return a darkened copy of *color*.

    Args:
        color: Source RGB tuple (0-255 per channel).
        factor: Multiplier applied to each channel (<1 darkens).

    Returns:
        Clamped RGB tuple.
    """
    return (
        max(0, int(color[0] * factor)),
        max(0, int(color[1] * factor)),
        max(0, int(color[2] * factor)),
    )


def make_groundhog_surface(
    color: tuple[int, int, int],
    height: int = 100,
) -> pygame.Surface:
    """Build a pygame SRCALPHA surface with a groundhog mascot in *color*.

    The mascot is drawn flat (no gradients, no shadows) using
    :mod:`pygame.draw` primitives scaled from a 90 x 110 baseline
    design.  The silhouette captures the key groundhog features:
    a wide flat head with large jowl pouches, prominent buck teeth,
    small ears, and a chunky squat body.

    Args:
        color: RGB fill colour for the groundhog body (0-255 per channel).
        height: Desired surface height in pixels (width scales
            proportionally from the 90 x 110 baseline).

    Returns:
        Transparent SRCALPHA pygame surface containing the groundhog.
    """
    W0, H0 = 90, 110  # baseline design dimensions
    scale = height / H0
    W = max(4, int(W0 * scale))
    H = max(4, height)

    surf = pygame.Surface((W, H), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))

    def s(v: float) -> int:
        return max(1, int(v * scale))

    cx = W // 2
    _DARK = (20, 20, 30)
    _WHITE = (245, 245, 250)
    _PINK = (220, 100, 100)
    belly = _lighten(color, 1.45)
    dark_fur = _darken(color, 0.65)

    # -------------------------------------------------------------------
    # Body — chunky squat oval sitting below the head
    # -------------------------------------------------------------------
    pygame.draw.ellipse(
        surf, color, pygame.Rect(cx - s(26), s(62), s(52), s(42))
    )

    # Belly patch (lighter oval on the front of the body)
    pygame.draw.ellipse(
        surf, belly, pygame.Rect(cx - s(13), s(68), s(26), s(28))
    )

    # Arms — short stubs raised to the sides
    pygame.draw.ellipse(
        surf, color, pygame.Rect(cx - s(42), s(65), s(18), s(22))
    )
    pygame.draw.ellipse(
        surf, color, pygame.Rect(cx + s(24), s(65), s(18), s(22))
    )

    # Feet — two small rounded blobs at the base
    pygame.draw.ellipse(
        surf, dark_fur, pygame.Rect(cx - s(22), s(98), s(18), s(10))
    )
    pygame.draw.ellipse(
        surf, dark_fur, pygame.Rect(cx + s(4), s(98), s(18), s(10))
    )

    # -------------------------------------------------------------------
    # Head — very wide, low profile, no neck
    # -------------------------------------------------------------------
    # Main head shape: wide flat ellipse
    pygame.draw.ellipse(
        surf, color, pygame.Rect(cx - s(32), s(18), s(64), s(50))
    )

    # Cheek jowls — wide pouches flaring out to each side
    pygame.draw.ellipse(
        surf, color, pygame.Rect(cx - s(44), s(30), s(28), s(34))
    )
    pygame.draw.ellipse(
        surf, color, pygame.Rect(cx + s(16), s(30), s(28), s(34))
    )

    # -------------------------------------------------------------------
    # Ears — small rounded bumps at the top corners of the head
    # -------------------------------------------------------------------
    pygame.draw.ellipse(
        surf, color, pygame.Rect(cx - s(30), s(10), s(18), s(16))
    )
    pygame.draw.ellipse(
        surf, color, pygame.Rect(cx + s(12), s(10), s(18), s(16))
    )
    # Inner ear (slightly darker)
    pygame.draw.ellipse(
        surf, dark_fur, pygame.Rect(cx - s(27), s(12), s(11), s(9))
    )
    pygame.draw.ellipse(
        surf, dark_fur, pygame.Rect(cx + s(16), s(12), s(11), s(9))
    )

    # -------------------------------------------------------------------
    # Eyes — small dark circles with a white shine dot
    # -------------------------------------------------------------------
    eye_r = s(4)
    pygame.draw.circle(surf, _DARK, (cx - s(13), s(30)), eye_r)
    pygame.draw.circle(surf, _DARK, (cx + s(13), s(30)), eye_r)
    shine_r = max(1, s(2))
    pygame.draw.circle(surf, _WHITE, (cx - s(11), s(28)), shine_r)
    pygame.draw.circle(surf, _WHITE, (cx + s(15), s(28)), shine_r)

    # -------------------------------------------------------------------
    # Nose — small dark oval between the eyes
    # -------------------------------------------------------------------
    pygame.draw.ellipse(
        surf, _DARK, pygame.Rect(cx - s(5), s(37), s(10), s(7))
    )

    # -------------------------------------------------------------------
    # Open mouth with buck teeth — the defining groundhog feature
    # -------------------------------------------------------------------
    # Mouth cavity (dark oval)
    pygame.draw.ellipse(
        surf, _DARK, pygame.Rect(cx - s(18), s(46), s(36), s(20))
    )

    # Tongue (pink, inside mouth)
    pygame.draw.ellipse(
        surf, _PINK, pygame.Rect(cx - s(10), s(52), s(20), s(10))
    )

    # Buck teeth (two large white rectangles side by side)
    pygame.draw.rect(
        surf,
        _WHITE,
        pygame.Rect(cx - s(12), s(46), s(10), s(14)),
        border_radius=s(2),
    )
    pygame.draw.rect(
        surf,
        _WHITE,
        pygame.Rect(cx + s(2), s(46), s(10), s(14)),
        border_radius=s(2),
    )
    # Tooth divider line
    pygame.draw.line(surf, _DARK, (cx, s(46)), (cx, s(59)), max(1, s(1)))

    return surf
