"""Render the web set: the gallery drawings, without the poster.

Each file is one plate from :mod:`illustration.plates` with the type,
the side chart and the wide bloom left out. The mark is the three-arc
square on the same two grounds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.image as mpimage
from illustration.plates import PLATES as GALLERY_PLATES
from illustration.solution import Solution
from illustration.theme import THEMES
from illustration.world import build_world

from . import draw
from .names import GROUNDS, OG_GROUND, OG_NAME, OG_PLATE, PLATES

_MODULES = {key.split("_", 1)[1]: module for key, module in GALLERY_PLATES}


def render_release(
    output: Path,
    grounds: Optional[Sequence[str]] = None,
    plates: Optional[Sequence[str]] = None,
    dpi: int = 120,
    width_in: float = 16.0,
) -> None:
    """Render the requested plates into *output*.

    Gallery plates are drawn with ``quiet=True``. The Open Graph crop
    is written when the nocturne field plate is rendered.

    Args:
        output: Directory receiving ``<plate>-<ground>.png``.
        grounds: Subset of ``nocturne`` and ``atlas``. Defaults to both.
        plates: Subset of plate names. Defaults to every plate.
        dpi: Raster resolution. 120 on a 16-inch figure is 1920×1080.
        width_in: Figure width in inches for the 16:9 plates.

    Raises:
        KeyError: If a ground or plate name is unknown.
    """
    chosen_grounds = tuple(grounds or GROUNDS)
    unknown_grounds = [name for name in chosen_grounds if name not in GROUNDS]
    if unknown_grounds:
        raise KeyError(f"unknown ground: {unknown_grounds[0]}")
    chosen_plates = tuple(plates or PLATES)
    unknown = [name for name in chosen_plates if name not in PLATES]
    if unknown:
        raise KeyError(f"unknown plate: {unknown[0]}")

    output.mkdir(parents=True, exist_ok=True)
    world = None
    solution = None
    if any(name != "mark" for name in chosen_plates):
        world = build_world()
        solution = Solution(world)

    for ground_name in chosen_grounds:
        theme = THEMES[ground_name]
        for plate in chosen_plates:
            target = output / f"{plate}-{ground_name}.png"
            if plate == "mark":
                draw.render_mark(theme, target, dpi)
            else:
                assert world is not None and solution is not None
                _MODULES[plate].render(
                    theme,
                    world,
                    solution,
                    target,
                    width_in=width_in,
                    dpi=dpi,
                    quiet=True,
                )
            print(f"wrote {target}")
            if plate == OG_PLATE and ground_name == OG_GROUND:
                image = mpimage.imread(target)
                og = output / OG_NAME
                mpimage.imsave(og, draw.og_crop(image))
                print(f"wrote {og}")
