"""Base graph data structure, re-exported from the compiled extension.

The implementation is the compiled ``Graph`` of :mod:`arco._arco`. On the
Rust side no struct carries it: deviation A-03 in
``docs/rust/DEVIATIONS.md`` records that the search surface the Python
hierarchy shared through this base became the ``DiscreteMap`` trait of
``arco-core``, and the binding layer re-creates the base class so the
``isinstance`` relationships Python callers relied on still hold.
"""

from arco._arco import Graph

__all__ = ["Graph"]
