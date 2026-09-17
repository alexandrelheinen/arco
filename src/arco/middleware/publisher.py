"""The publisher mixin, re-exported from the compiled extension.

The implementation is the node publisher in the ``arco-runtime`` crate, registered back
under its Python spelling by the binding layer and reaching callers
through :mod:`arco._arco`.
"""

from arco._arco import BusPublisher

__all__ = ["BusPublisher"]
