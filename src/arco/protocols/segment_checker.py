"""SegmentChecker: duck-typed contract for edge collision checks."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SegmentChecker(Protocol):
    """Collision checker for a straight segment between two states.

    ``__call__(a, b) → bool`` returns True when the segment is free.
    """

    def __call__(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Return True if the segment from *a* to *b* is collision-free."""
