"""PrunerLike: duck-typed contract for pipeline pruners."""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class PrunerLike(Protocol):
    """Pruner stage: ``prune(path) → shortened path``."""

    def prune(self, path: List[np.ndarray]) -> List[np.ndarray]:
        """Return a shortened collision-free path."""
