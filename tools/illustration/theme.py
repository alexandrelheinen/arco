"""Theme: the visual language shared by every ARCO gallery plate.

A :class:`Theme` bundles every colour, alpha ramp and type size used by
the gallery renderer, so a plate module never hardcodes a hex value.  Two
themes ship:

``nocturne``
    Deep-navy stage with neon bloom.  Screen, slide decks, web headers.
``atlas``
    Warm paper stage with ink strokes and soft halos.  Print and journal
    figures, where a dark plate wastes toner and loses fine branches.

Algorithm hues come from :mod:`arco.config.palette` so the gallery and
the ``arcosim`` renderer stay the same family of blue / green / violet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

from matplotlib import font_manager

from arco.config.palette import method_base_hex, obstacle_hex

# (width multiplier, alpha) passes drawn bottom-up under every stroke.
GlowRamp = Tuple[Tuple[float, float], ...]

FONT_TEXT = "DejaVu Sans"
FONT_MONO = "DejaVu Sans Mono"

# Display type wants a lighter weight than the one matplotlib bundles.
# The file below ships with the standard DejaVu package; when it is not
# installed the gallery silently falls back to the regular weight.
_LIGHT_CANDIDATES = (
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-ExtraLight.ttf"),
    Path("/usr/share/fonts/TTF/DejaVuSans-ExtraLight.ttf"),
)


def _light_face_file():
    """Return the path of an installed light display face, if any.

    Returns:
        Path to the font file, or ``None`` when none is installed.
    """
    for candidate in _LIGHT_CANDIDATES:
        if candidate.exists():
            font_manager.fontManager.addfont(str(candidate))
            return candidate
    return None


LIGHT_FACE_FILE = _light_face_file()


def display_font(size: float) -> font_manager.FontProperties:
    """Return the font used for titles and the wordmark.

    Args:
        size: Point size.

    Returns:
        Font properties for light-weight display type, falling back to
        the regular text face when no light face is installed.
    """
    if LIGHT_FACE_FILE is not None:
        return font_manager.FontProperties(
            fname=str(LIGHT_FACE_FILE), size=size
        )
    return font_manager.FontProperties(family=FONT_TEXT, size=size)


def text_font(size: float) -> font_manager.FontProperties:
    """Return the font used for labels and legend entries.

    Args:
        size: Point size.

    Returns:
        Font properties for body type.
    """
    return font_manager.FontProperties(family=FONT_TEXT, size=size)


def mono_font(size: float) -> font_manager.FontProperties:
    """Return the font used for the run-metrics block.

    Args:
        size: Point size.

    Returns:
        Font properties for monospace type.
    """
    return font_manager.FontProperties(family=FONT_MONO, size=size)


@dataclass(frozen=True)
class Theme:
    """A complete visual configuration for one gallery plate rendering.

    Attributes:
        name: Theme identifier used in output paths (``nocturne`` / ``atlas``).
        background: Outer canvas colour, also the saved PNG matte.
        background_core: Colour at the centre of the stage gradient.
        grain: Alpha of the film-grain overlay (0 disables it).
        rule: Hairline colour for grids, frames and separators.
        text: Primary type colour (titles, wordmark).
        text_dim: Secondary type colour (captions, metrics).
        obstacle_fill: Interior colour of obstacle bodies.
        obstacle_edge: Rim colour of obstacle bodies.
        obstacle_alpha: Interior alpha of obstacle bodies.
        rrt: RRT* hue.
        sst: SST hue.
        astar: A* hue.
        accent: Warm highlight used for executed motion and markers.
        ice: Cool highlight used for the bright end of cost ramps.
        glow: Bloom ramp applied under emphasised strokes.
        glow_soft: Lighter bloom ramp for dense line collections.
        tree_alpha: Alpha at the root end of a planner tree.
        tree_alpha_tip: Alpha at the leaf end of a planner tree.
        title_size: Point size of a plate title.
        subtitle_size: Point size of the one-line plate subtitle.
        caption_size: Point size of the monospace metrics line.
        wordmark_size: Point size of the ``ARCO`` wordmark.
    """

    name: str
    background: str
    background_core: str
    grain: float
    rule: str
    text: str
    text_dim: str
    obstacle_fill: str
    obstacle_edge: str
    obstacle_alpha: float
    rrt: str
    sst: str
    astar: str
    accent: str
    ice: str
    glow: GlowRamp
    glow_soft: GlowRamp
    tree_alpha: float
    tree_alpha_tip: float
    title_size: float = 21.0
    subtitle_size: float = 12.0
    caption_size: float = 10.0
    wordmark_size: float = 15.0
    method_colors: dict = field(default_factory=dict)

    def method(self, key: str) -> str:
        """Return the hue for an algorithm key.

        Args:
            key: One of ``"rrt"``, ``"sst"`` or ``"astar"``.

        Returns:
            Hex colour string for that algorithm in this theme.

        Raises:
            KeyError: If *key* names no algorithm in the theme.
        """
        return {"rrt": self.rrt, "sst": self.sst, "astar": self.astar}[key]


NOCTURNE = Theme(
    name="nocturne",
    background="#05070d",
    background_core="#141d33",
    grain=0.030,
    rule="#26304a",
    text="#e8ecf6",
    text_dim="#78859f",
    obstacle_fill="#171420",
    obstacle_edge=obstacle_hex(),
    obstacle_alpha=0.92,
    rrt="#5b93ef",
    sst="#3fc984",
    astar="#a97bf2",
    accent="#ffc46b",
    ice="#bfe4ff",
    glow=((9.0, 0.028), (5.0, 0.055), (2.6, 0.105), (1.0, 1.0)),
    glow_soft=((5.0, 0.030), (2.4, 0.070), (1.0, 0.95)),
    tree_alpha=0.10,
    tree_alpha_tip=0.55,
)

ATLAS = Theme(
    name="atlas",
    background="#f4f1ea",
    background_core="#fffdf8",
    grain=0.022,
    rule="#cdc6b8",
    text="#1b1c22",
    text_dim="#6d6a62",
    obstacle_fill="#ded5c6",
    obstacle_edge="#b4544f",
    obstacle_alpha=0.95,
    rrt=method_base_hex("rrt"),
    sst="#2f8f56",
    astar=method_base_hex("astar"),
    accent="#c2700f",
    ice="#2b6ca8",
    glow=((6.5, 0.045), (3.0, 0.100), (1.0, 1.0)),
    glow_soft=((3.4, 0.055), (1.0, 0.95)),
    tree_alpha=0.16,
    tree_alpha_tip=0.70,
)

THEMES = {NOCTURNE.name: NOCTURNE, ATLAS.name: ATLAS}
