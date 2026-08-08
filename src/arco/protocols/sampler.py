"""Sampler: duck-typed contract for continuous planner sampling."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Sampler(Protocol):
    """Random-state sampler used by RRT*/SST.

    ``__call__(rng) → state`` matches the default uniform AABB sampler.
    """

    def __call__(self, rng: np.random.Generator) -> np.ndarray:
        """Return a random state sample."""
