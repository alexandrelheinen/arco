"""Filenames published with a GitHub release.

One list, so the renderer, the publish script and the tests cannot
drift apart. ``arc-og.png`` is the center crop of ``arc-dark.png``,
not a separate composition.
"""

from __future__ import annotations

from typing import Tuple

PLATES: Tuple[str, ...] = (
    "mark",
    "arc",
    "branch",
    "flood",
    "pair",
    "track",
    "ribbon",
)
GROUNDS: Tuple[str, ...] = ("dark", "light")
OG_NAME = "arc-og.png"


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
