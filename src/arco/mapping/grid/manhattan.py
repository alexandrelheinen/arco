"""Manhattan grid, re-exported from the compiled extension.

The implementation is ``ManhattanGrid`` in the ``arco-mapping`` crate,
reaching Python through :mod:`arco._arco`. Four-connected neighbors and
L1 distance come as one pairing fixed at construction rather than as an
overridable method, per deviation A-03 in ``docs/rust/DEVIATIONS.md``.
"""

from arco._arco import ManhattanGrid

__all__ = ["ManhattanGrid"]
