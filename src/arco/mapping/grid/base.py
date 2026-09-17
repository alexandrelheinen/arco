"""Base N-dimensional grid, re-exported from the compiled extension.

The implementation is ``GridCells`` in the ``arco-mapping`` crate, which
owns the extent, the cell size and the cell states, and reaches Python
through :mod:`arco._arco` as the shared base of the two grid metrics.
Constructing it directly raises :class:`TypeError`, because the crate
fixes the metric and the neighborhood together at construction and only
:class:`~arco.mapping.grid.manhattan.ManhattanGrid` or
:class:`~arco.mapping.grid.euclidean.EuclideanGrid` names a valid
pairing. Deviations A-03 and A-23 in ``docs/rust/DEVIATIONS.md`` record
both points.
"""

from arco._arco import Grid

__all__ = ["Grid"]
