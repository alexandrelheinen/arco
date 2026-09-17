"""Controller: abstract base for feedback controllers, from the extension.

The implementation is ``PyController`` in the ``arco-py`` binding layer,
wrapping the controller role that ``arco-control`` carries in
``PidController`` and the ``PathTracker`` trait, and reaching callers
through :mod:`arco._arco`. The compiled class keeps the abstract-base
semantics of the Python version: ``control`` stays abstract, so
constructing the base raises :class:`TypeError` and a subclass has to
implement it. Deviation A-24: an override reached from inside a loop
crosses the interpreter lock on every call.
"""

from arco._arco import Controller

__all__ = ["Controller"]
