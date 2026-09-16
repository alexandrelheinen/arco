"""Plate modules, in gallery order.

Each module exposes ``TITLE``, ``SUBTITLE`` and a ``render`` function
with the same signature, so the CLI can treat them uniformly.
"""

from __future__ import annotations

from . import contest, field, growth, pursuit, reachability, refine, wavefront

PLATES = (
    ("01_field", field),
    ("02_wavefront", wavefront),
    ("03_growth", growth),
    ("04_contest", contest),
    ("05_refine", refine),
    ("06_pursuit", pursuit),
    ("07_reachability", reachability),
)

__all__ = [
    "PLATES",
    "contest",
    "field",
    "growth",
    "pursuit",
    "reachability",
    "refine",
    "wavefront",
]
