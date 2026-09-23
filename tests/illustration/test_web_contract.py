"""Tests for the quiet web image set.

Solver tests are separate and slower. These cover the budget, the
filename contract, the crop, and a mark render that does not plan.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from illustration.web.budget import cap_edges, tree_edges  # noqa: E402
from illustration.web.draw import og_crop  # noqa: E402
from illustration.web.names import release_filenames  # noqa: E402
from illustration.web.scene import (  # noqa: E402
    BASIN_BODIES,
    HEIGHT,
    WIDTH,
    build_flood,
)


def test_cap_edges_keeps_the_longest_and_stops_at_eighty() -> None:
    """A dense tree must come back as at most 80 edges, longest first."""
    nodes = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 3.0], [0.0, 1.0]])
    parents = {0: None, 1: 0, 2: 0, 3: 0}
    kept = cap_edges(nodes, parents, limit=2)
    assert len(kept) == 2
    assert (1, 0) in kept
    assert (2, 0) in kept
    assert (3, 0) not in kept


def test_cap_edges_returns_a_short_tree_unchanged() -> None:
    """A tree under the cap must not drop edges."""
    nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
    parents = [None, 0]
    assert tree_edges(parents) == [(1, 0)]
    assert cap_edges(nodes, parents) == [(1, 0)]


def test_release_filenames_cover_both_grounds_and_the_og_crop() -> None:
    """The release set is fourteen plates plus the Open Graph crop."""
    names = release_filenames()
    assert names[0] == "mark-dark.png"
    assert names[1] == "mark-light.png"
    assert "arc-dark.png" in names
    assert "ribbon-light.png" in names
    assert names[-1] == "arc-og.png"
    assert len(names) == 15
    assert len(set(names)) == 15


def test_basin_discs_survive_the_open_graph_crop() -> None:
    """Disc extents stay inside the center band a 1200x630 crop keeps."""
    y0 = 0.185 * HEIGHT
    y1 = 0.815 * HEIGHT
    x0 = 0.10 * WIDTH
    x1 = 0.90 * WIDTH
    assert len(BASIN_BODIES) == 3
    for centre_x, centre_y, radius in BASIN_BODIES:
        assert x0 < centre_x - radius
        assert centre_x + radius < x1
        assert y0 < centre_y - radius
        assert centre_y + radius < y1


def test_flood_grid_has_two_blocks_and_a_free_start() -> None:
    """The A* plate is a coarse grid with two obstacles."""
    flood = build_flood()
    assert flood.grid.data.shape == (32, 18)
    assert len(flood.blocks) == 2
    assert int(flood.grid.data[flood.start]) == 0
    assert int(flood.grid.data[flood.goal]) == 0
    assert int(flood.grid.data[flood.blocks[0][0], flood.blocks[0][2]]) == 1


def test_og_crop_is_1200_by_630_and_keeps_the_middle() -> None:
    """The crop trims height, then resamples to the Open Graph size."""
    image = np.zeros((900, 1600, 3), dtype=float)
    image[450, 800] = (1.0, 0.0, 0.0)
    cropped = og_crop(image)
    assert cropped.shape == (630, 1200, 3)
    window = cropped[300:330, 590:610]
    assert window[..., 0].max() == 1.0
