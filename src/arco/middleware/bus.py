"""The typed in-process bus, re-exported from the compiled extension.

The implementation is ``Bus`` in the ``arco-runtime`` crate, registered back
under its Python spelling by the binding layer and reaching callers
through :mod:`arco._arco`. Routing is by the frame's Python class rather than by a Rust type, because a frame type is chosen at run time; `subscribe` still hands back a `queue.Queue` that callers poll with `get_nowait`.
"""

from arco._arco import Bus, InMemoryBus

__all__ = ["Bus", "InMemoryBus"]
