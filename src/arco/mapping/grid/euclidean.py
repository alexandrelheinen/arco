"""Euclidean grid, re-exported from the compiled extension.

The implementation is ``EuclideanGrid`` in the ``arco-mapping`` crate,
reaching Python through :mod:`arco._arco`. Eight-connected neighbors and
straight-line distance come as one pairing fixed at construction rather
than as an overridable method, per deviation A-03 in
``docs/rust/DEVIATIONS.md``.
"""

from arco._arco import EuclideanGrid

__all__ = ["EuclideanGrid"]
