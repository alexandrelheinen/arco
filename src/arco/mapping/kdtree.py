"""KD-tree occupancy map, re-exported from the compiled extension.

The implementation is ``KdTreeOccupancy`` in the ``arco-mapping`` crate,
registered back under its Python spelling by the binding layer and
reaching callers through :mod:`arco._arco`. The abstract base it used to
inherit is the ``Occupancy`` trait of ``arco-core`` on the Rust side, so
:mod:`arco.mapping.occupancy` keeps carrying the Python version of that
base for callers who subclass it.
"""

from arco._arco import KDTreeOccupancy

__all__ = ["KDTreeOccupancy"]
