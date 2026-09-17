"""The Dubins primitive, re-exported from the compiled extension.

The implementation is ``DubinsPrimitive`` in the ``arco-guidance`` crate, registered
back under its Python spelling by the binding layer and reaching callers
through :mod:`arco._arco`. Deviation C-12: `steer` returns only the two endpoints, in Python as well.
"""

from arco._arco import DubinsPrimitive

__all__ = ["DubinsPrimitive"]
