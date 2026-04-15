"""Shared visualization utilities for ARCO example and simulation tools."""

from __future__ import annotations

import numpy as np


def parent_dict_to_list(
    parent: dict[int, int | None], n: int
) -> list[int]:
    """Convert a planner parent dict to a parallel index list.

    Args:
        parent: Dict mapping node index to parent index (``None`` for root).
        n: Total number of nodes.

    Returns:
        List of length *n* with ``-1`` for root and non-negative parent
        indices for all other nodes.
    """
    return [
        -1 if parent.get(i) is None else int(parent[i])  # type: ignore[arg-type]
        for i in range(n)
    ]


def polyline_length(path: list[np.ndarray] | None) -> float:
    """Return total Euclidean arc length for a waypoint sequence.

    Args:
        path: Sequence of position arrays, or ``None`` for an empty path.

    Returns:
        Total arc length in the same units as the coordinates.  Returns
        ``0.0`` when *path* is ``None`` or has fewer than two points.
    """
    if path is None or len(path) < 2:
        return 0.0
    return sum(
        float(np.linalg.norm(path[i + 1] - path[i]))
        for i in range(len(path) - 1)
    )


def format_clock(seconds: float) -> str:
    """Format a duration in seconds as ``MMminSSs``.

    Args:
        seconds: Duration in seconds (non-negative).

    Returns:
        Human-readable string such as ``"02min07s"``.
    """
    rounded = int(round(max(0.0, seconds)))
    mins, secs = divmod(rounded, 60)
    return f"{mins:02d}min{secs:02d}s"
