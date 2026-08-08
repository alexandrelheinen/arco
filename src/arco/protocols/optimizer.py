"""OptimizerLike: duck-typed contract for pipeline optimizers."""

from __future__ import annotations

from typing import Any, List, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class OptimizerLike(Protocol):
    """Optimizer stage: ``optimize(path) → result``."""

    def optimize(self, path: List[np.ndarray]) -> Any:
        """Refine *path* into a time-parameterized trajectory result."""
