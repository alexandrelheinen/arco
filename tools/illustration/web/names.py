"""Filenames published with a GitHub release.

One list, so the renderer, the publish script and the tests cannot
drift apart. The plates are the gallery drawings. ``field-og.png`` is
the center crop of ``field-nocturne.png``, not a separate composition.
"""

from __future__ import annotations

from typing import Tuple

PLATES: Tuple[str, ...] = (
    "mark",
    "field",
    "wavefront",
    "growth",
    "contest",
    "refine",
    "pursuit",
    "reachability",
)
GROUNDS: Tuple[str, ...] = ("nocturne", "atlas")
OG_PLATE = "field"
OG_GROUND = "nocturne"
OG_NAME = "field-og.png"


def plate_filenames() -> Tuple[str, ...]:
    """Return the fourteen plate files, dark ground then light per plate.

    Returns:
        ``<plate>-<ground>.png`` for every plate and both grounds.
    """
    return tuple(
        f"{plate}-{ground}.png" for plate in PLATES for ground in GROUNDS
    )


def release_filenames() -> Tuple[str, ...]:
    """Return every file a release receives, plates plus the OG crop.

    Returns:
        Plate filenames followed by :data:`OG_NAME`.
    """
    return plate_filenames() + (OG_NAME,)
