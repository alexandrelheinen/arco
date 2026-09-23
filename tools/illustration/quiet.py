"""The web export of a gallery plate: the picture, without the poster.

A quiet plate keeps the solver drawing (trees, expansion fields, the
routes they return) and drops the three things that do not belong in a
figure sitting next to a paragraph: type, a side chart, and the wide
bloom under a stroke.
"""

from __future__ import annotations

from dataclasses import replace

from .theme import Theme

# Two passes. The wide one is a short, faint rim; the core is the stroke.
SOFT_GLOW = ((2.2, 0.10), (1.0, 0.70))
SOFT_GLOW_SOFT = ((1.7, 0.08), (1.0, 0.50))
FULL_BLEED = (0.0, 0.0, 1.0, 1.0)


def soften(theme: Theme) -> Theme:
    """Return *theme* with the bloom pulled in.

    Args:
        theme: Gallery theme, untouched when the caller wants the poster.

    Returns:
        A copy whose glow ramps no longer throw a wide lamp under a curve.
    """
    return replace(theme, glow=SOFT_GLOW, glow_soft=SOFT_GLOW_SOFT)
