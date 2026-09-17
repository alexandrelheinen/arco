"""Route planning over a positioned graph, re-exported from the compiled extension.

The implementation is ``RouteRouter`` in the ``arco-planning`` crate, registered back
under its Python spelling by the binding layer and reaching callers
through :mod:`arco._arco`. ``RouteResult`` stays a ``namedtuple``, so positional unpacking, indexing and tuple equality behave as they always did.
"""

from arco._arco import RouteResult, RouteRouter

__all__ = ["RouteResult", "RouteRouter"]
