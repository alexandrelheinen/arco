"""Render the mark, which does not need a planner."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip(
    "matplotlib",
    reason="the mark render needs matplotlib, a dev extra",
)
import matplotlib.image as mpimage  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

from illustration.plates import PLATES as GALLERY_PLATES  # noqa: E402
from illustration.quiet import SOFT_GLOW, soften  # noqa: E402
from illustration.theme import NOCTURNE  # noqa: E402
from illustration.web.draw import render_mark  # noqa: E402
from illustration.web.names import PLATES  # noqa: E402
from illustration.web.theme import DARK, LIGHT  # noqa: E402


def test_mark_is_a_square_png_on_the_dark_ground(tmp_path: Path) -> None:
    """The mark is 512 px, flat, and carries no type."""
    target = tmp_path / "mark-dark.png"
    render_mark(DARK, target)
    image = mpimage.imread(target)
    assert image.shape[0] == 512
    assert image.shape[1] == 512
    # Corner pixel is the ground, so the canvas is not a poster frame.
    corner = image[0, 0, :3]
    assert corner.max() < 0.15


def test_light_mark_uses_the_paper_ground(tmp_path: Path) -> None:
    """Traço is the same mark on the light ground."""
    target = tmp_path / "mark-light.png"
    render_mark(LIGHT, target)
    image = mpimage.imread(target)
    corner = image[0, 0, :3]
    assert corner.min() > 0.9
    assert DARK.rrt == LIGHT.rrt == "#4477CC"
    assert DARK.sst == "#44AA66"
    assert DARK.astar == "#7744BB"


def test_release_plates_are_the_gallery_drawings() -> None:
    """Every gallery plate is published; the mark is the extra square."""
    stems = {key.split("_", 1)[1] for key, _module in GALLERY_PLATES}
    assert stems <= set(PLATES)
    assert "mark" in PLATES
    assert "arc" not in PLATES


def test_soften_pulls_the_bloom_in() -> None:
    """Quiet plates keep the hue and drop the wide lamp under a stroke."""
    quiet = soften(NOCTURNE)
    assert quiet.glow == SOFT_GLOW
    assert quiet.glow[0][0] < NOCTURNE.glow[0][0]
    assert quiet.rrt == NOCTURNE.rrt
    assert quiet is not NOCTURNE
