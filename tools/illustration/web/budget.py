"""How much of a planner tree a thumbnail is allowed to draw.

A full RRT* tree turns into texture at card size. The web plates keep
the longest edges, which are the ones that still separate when the
image is shown at about 320 px.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

MAX_TREE_EDGES = 80

Parents = Union[Dict[int, Optional[int]], Sequence[Optional[int]]]


def tree_edges(parents: Parents) -> List[Tuple[int, int]]:
    """Return ``(child, parent)`` pairs, skipping the root.

    Args:
        parents: Parent index per node. A ``dict`` or a sequence. The
            root stores ``None``.

    Returns:
        One pair per edge. Indices are plain ``int``.
    """
    if hasattr(parents, "items"):
        items = parents.items()
    else:
        items = enumerate(parents)
    edges: List[Tuple[int, int]] = []
    for child, parent in items:
        if parent is None:
            continue
        edges.append((int(child), int(parent)))
    return edges


def cap_edges(
    nodes: np.ndarray,
    parents: Parents,
    limit: int = MAX_TREE_EDGES,
) -> List[Tuple[int, int]]:
    """Keep at most *limit* edges, preferring the longest.

    Args:
        nodes: ``(N, 2)`` node positions in world units.
        parents: Parent index per node.
        limit: Maximum number of edges to return.

    Returns:
        The kept ``(child, parent)`` pairs. Fewer than *limit* when the
        tree itself is smaller.

    Raises:
        ValueError: If *limit* is negative.
    """
    if limit < 0:
        raise ValueError("limit must be non-negative")
    edges = tree_edges(parents)
    if len(edges) <= limit:
        return edges
    ranked = sorted(edges, key=lambda edge: _length(nodes, edge), reverse=True)
    return ranked[:limit]


def _length(nodes: np.ndarray, edge: Tuple[int, int]) -> float:
    """Return the Euclidean length of one tree edge.

    Args:
        nodes: ``(N, 2)`` node positions.
        edge: ``(child, parent)`` indices.

    Returns:
        Length in world units.
    """
    child, parent = edge
    delta = np.asarray(nodes[child], dtype=float) - np.asarray(
        nodes[parent], dtype=float
    )
    return float(np.hypot(delta[0], delta[1]))
