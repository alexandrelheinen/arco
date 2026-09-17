"""The subscriber mixin, re-exported from the compiled extension.

The implementation is the node subscriber in the ``arco-runtime`` crate, registered back
under its Python spelling by the binding layer and reaching callers
through :mod:`arco._arco`.
"""

from arco._arco import BusSubscriber

__all__ = ["BusSubscriber"]
