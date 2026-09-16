"""Colour ramps used to encode scalar fields on gallery plates.

Each ramp is a function of the active :class:`~illustration.theme.Theme`
so the same encoding reads correctly on the dark and the light stage: on
``nocturne`` the bright end is light-emitting, on ``atlas`` it is ink.
"""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap

from .theme import Theme


def _ramp(name: str, colors) -> LinearSegmentedColormap:
    """Build a named linear colormap.

    Args:
        name: Colormap name.
        colors: Ordered colour stops.

    Returns:
        The assembled colormap.
    """
    return LinearSegmentedColormap.from_list(name, list(colors), N=512)


def cost(theme: Theme) -> LinearSegmentedColormap:
    """Return the ramp for cost-to-come on a sampling tree.

    Args:
        theme: Active theme.

    Returns:
        Colormap from root (dark) to frontier (bright).
    """
    if theme.name == "nocturne":
        return _ramp("arco_cost", ["#1b2a5e", theme.rrt, theme.ice])
    return _ramp("arco_cost", ["#0f2b4d", theme.rrt, "#8fb8e8"])


def wavefront(theme: Theme) -> LinearSegmentedColormap:
    """Return the ramp for A* expansion order.

    Args:
        theme: Active theme.

    Returns:
        Colormap from first expansion to last.
    """
    if theme.name == "nocturne":
        return _ramp(
            "arco_wave",
            ["#241543", "#5c2f9e", theme.astar, "#d79bf0", "#f3d4ff"],
        )
    return _ramp("arco_wave", ["#efe6f7", "#b99ae0", theme.astar, "#3d1f6b"])


def speed(theme: Theme) -> LinearSegmentedColormap:
    """Return the ramp for trajectory speed.

    Args:
        theme: Active theme.

    Returns:
        Colormap from slow to fast.
    """
    if theme.name == "nocturne":
        return _ramp(
            "arco_speed",
            ["#3f6fd8", "#8a5bd0", "#e0603f", theme.accent, "#ffeec4"],
        )
    return _ramp(
        "arco_speed",
        ["#2f4f9e", "#6a5bb0", "#c2700f", "#9a3f14", "#5e1f0a"],
    )


def error(theme: Theme) -> LinearSegmentedColormap:
    """Return the diverging ramp for signed lateral error.

    Args:
        theme: Active theme.

    Returns:
        Colormap from one side of the reference to the other.
    """
    if theme.name == "nocturne":
        return _ramp(
            "arco_error",
            [theme.rrt, "#8fa8c8", "#f2f5ff", "#ffb27a", "#ff6b4a"],
        )
    return _ramp(
        "arco_error", [theme.rrt, "#8fa8c8", "#efe9dd", "#d99b4a", "#a33a22"]
    )
