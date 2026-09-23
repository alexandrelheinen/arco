"""The two flat grounds the web set is drawn on.

``dark`` is Sinal, for thumbnails and Open Graph. ``light`` is Traço,
for a figure on a light page. Algorithm hues come from
``colors.yml`` through :func:`arco.config.palette.method_base_hex`, the
same colors the simulator uses. Ground colors are part of this set and
are not the simulator's obstacle red.
"""

from __future__ import annotations

from dataclasses import dataclass

from arco.config.palette import method_base_hex


@dataclass(frozen=True)
class Ground:
    """Flat colors and the tree alpha for one web ground.

    Attributes:
        name: ``dark`` or ``light``, used in the filename.
        background: Canvas color.
        obstacle: Fill of a body or a blocked cell.
        ink: Stroke used for an answer that is not an algorithm hue.
        ink_dim: Quieter ink, used for a reference curve.
        rrt: RRT* hue.
        sst: SST hue.
        astar: A* hue.
        tree_alpha: Opacity of the exploration tree.
        tree_uses_ink: When true the tree is ink; otherwise it is the
            RRT* hue. Sinal keeps the tree in blue, Traço in ink.
    """

    name: str
    background: str
    obstacle: str
    ink: str
    ink_dim: str
    rrt: str
    sst: str
    astar: str
    tree_alpha: float
    tree_uses_ink: bool


DARK = Ground(
    name="dark",
    background="#0c1016",
    obstacle="#1a2030",
    ink="#e7ebf2",
    ink_dim="#8b95a8",
    rrt=method_base_hex("rrt"),
    sst=method_base_hex("sst"),
    astar=method_base_hex("astar"),
    tree_alpha=0.20,
    tree_uses_ink=False,
)

LIGHT = Ground(
    name="light",
    background="#f6f4ef",
    obstacle="#e4dfd6",
    ink="#1c1e24",
    ink_dim="#6e6a62",
    rrt=method_base_hex("rrt"),
    sst=method_base_hex("sst"),
    astar=method_base_hex("astar"),
    tree_alpha=0.35,
    tree_uses_ink=True,
)

GROUNDS = {DARK.name: DARK, LIGHT.name: LIGHT}
