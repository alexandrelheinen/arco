"""Weighted undirected graph, re-exported from the compiled extension.

The implementation is ``WeightedGraph`` in the ``arco-mapping`` crate,
reaching Python through :mod:`arco._arco`. It owns the adjacency and the
edge weights rather than inheriting them, per deviation A-03 in
``docs/rust/DEVIATIONS.md``.
"""

from arco._arco import WeightedGraph

__all__ = ["WeightedGraph"]
