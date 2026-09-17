"""RigidBody: abstract planar body, from the compiled extension.

The implementation is the ``RigidBody`` trait of the ``arco-control``
crate together with its ``BodyState``, registered back under the Python
spelling by the binding layer and reaching callers through
:mod:`arco._arco`. The compiled class keeps the abstract-base semantics of
the Python version: ``inertia`` and ``bounding_radius`` stay abstract, so
constructing the base raises :class:`TypeError` and a subclass has to
supply both. Deviation A-24: a subclass written in Python crosses the
interpreter lock every time one of those is read.
"""

from arco._arco import RigidBody

__all__ = ["RigidBody"]
