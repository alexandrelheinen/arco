"""Render the web set to a directory of PNG files."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.image as mpimage

from . import draw
from .names import OG_NAME, PLATES
from .scene import build_basin, build_flood
from .solve import load_basin, load_flood
from .theme import GROUNDS

_BASIN_PLATES = ("arc", "branch", "pair", "track", "ribbon")


def render_release(
    output: Path,
    grounds: Optional[Sequence[str]] = None,
    plates: Optional[Sequence[str]] = None,
    rrt_samples: Optional[int] = None,
    sst_samples: Optional[int] = None,
    dpi: int = draw.DPI,
) -> None:
    """Render the requested plates into *output*.

    The Open Graph crop is written when ``arc`` is rendered on the dark
    ground.

    Args:
        output: Directory receiving ``<plate>-<ground>.png``.
        grounds: Subset of ``dark`` and ``light``. Defaults to both.
        plates: Subset of plate names. Defaults to every plate.
        rrt_samples: RRT* budget. Defaults to the release budget.
        sst_samples: SST budget. Defaults to the release budget.
        dpi: Raster resolution of the masters.

    Raises:
        KeyError: If a ground or plate name is unknown.
    """
    chosen_grounds = tuple(
        GROUNDS[name] for name in (grounds or tuple(GROUNDS))
    )
    chosen_plates = tuple(plates or PLATES)
    unknown = [name for name in chosen_plates if name not in PLATES]
    if unknown:
        raise KeyError(f"unknown plate: {unknown[0]}")

    output.mkdir(parents=True, exist_ok=True)
    basin_runs = None
    flood_run = None
    basin = None
    flood = None
    if any(name in _BASIN_PLATES for name in chosen_plates):
        basin = build_basin()
        kwargs = {}
        if rrt_samples is not None:
            kwargs["rrt_samples"] = rrt_samples
        if sst_samples is not None:
            kwargs["sst_samples"] = sst_samples
        basin_runs = load_basin(basin, **kwargs)
    if "flood" in chosen_plates:
        flood = build_flood()
        flood_run = load_flood(flood)

    for ground in chosen_grounds:
        for plate in chosen_plates:
            target = output / f"{plate}-{ground.name}.png"
            _one(
                plate, ground, target, basin, basin_runs, flood, flood_run, dpi
            )
            print(f"wrote {target}")
            if plate == "arc" and ground.name == "dark":
                og = output / OG_NAME
                image = mpimage.imread(target)
                mpimage.imsave(og, draw.og_crop(image))
                print(f"wrote {og}")


def _one(
    plate, ground, target, basin, basin_runs, flood, flood_run, dpi
) -> None:
    """Dispatch one plate.

    Args:
        plate: Plate name.
        ground: Color ground.
        target: Destination PNG.
        basin: Basin scene, or ``None`` when unused.
        basin_runs: Basin solver output, or ``None``.
        flood: Flood scene, or ``None``.
        flood_run: Flood solver output, or ``None``.
        dpi: Raster resolution.
    """
    if plate == "mark":
        draw.render_mark(ground, target, dpi)
        return
    if plate == "arc":
        draw.render_arc(ground, basin, basin_runs, target, dpi)
        return
    if plate == "branch":
        draw.render_branch(ground, basin, basin_runs, target, dpi)
        return
    if plate == "pair":
        draw.render_pair(ground, basin, basin_runs, target, dpi)
        return
    if plate == "track":
        draw.render_track(ground, basin, basin_runs, target, dpi)
        return
    if plate == "ribbon":
        draw.render_ribbon(ground, basin, basin_runs, target, dpi)
        return
    if plate == "flood":
        draw.render_flood(ground, flood, flood_run, target, dpi)
        return
    raise KeyError(plate)
