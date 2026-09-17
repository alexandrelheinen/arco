"""Pure pursuit path tracking, re-exported from the compiled extension.

The implementation is ``PurePursuitTracker`` in the ``arco-control`` crate.
``PurePursuitController`` is that tracker registered under its Python
spelling, and ``_find_lookahead`` and ``_circle_segment_intersection`` are
the same crate's lookahead geometry, bound as free functions so the
regression tests that pin the off-track fallback keep calling them
directly. All three reach callers through :mod:`arco._arco`.
"""

from __future__ import annotations

from arco._arco import (
    PurePursuitController,
    _circle_segment_intersection,
    _find_lookahead,
)

__all__ = ["PurePursuitController"]
