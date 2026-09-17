"""Cartesian graph, re-exported from the compiled extension.

The implementation is ``CartesianGraph`` in the ``arco-mapping`` crate,
reaching Python through :mod:`arco._arco`. It owns a weighted graph plus
the node positions and delegates to it, which is what deviation A-03 in
``docs/rust/DEVIATIONS.md`` puts in place of the Python inheritance
chain.
"""

from arco._arco import CartesianGraph

__all__ = ["CartesianGraph"]
