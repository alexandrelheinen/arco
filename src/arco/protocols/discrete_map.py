"""DiscreteMap: duck-typed contract for discrete planners."""

from __future__ import annotations

from typing import Any, Iterator, Protocol, runtime_checkable


@runtime_checkable
class DiscreteMap(Protocol):
    """Map/graph surface expected by discrete planners (A*, route).

    Concrete types include grids and weighted/Cartesian graphs.  Optional
    members ``heuristic`` and ``is_occupied`` are detected via
    ``hasattr`` by planners when present.
    """

    def neighbors(self, node: Any) -> Iterator[Any]:
        """Yield adjacent nodes of *node*."""

    def distance(self, node_a: Any, node_b: Any) -> float:
        """Return the edge cost between two adjacent nodes."""
