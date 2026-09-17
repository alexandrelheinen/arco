"""Artificial potential field avoidance, from the compiled extension.

The implementation is ``ArtificialPotentialField`` in the ``arco-control``
crate, registered back under its Python spelling by the binding layer and
reaching callers through :mod:`arco._arco`. The class stays callable as
``field(x, y, theta)``, so it still satisfies the
:class:`~arco.protocols.avoidance.AvoidanceStrategy` protocol. Deviation
A-16: ``nearest_obstacle`` measures from the obstacle surface.
"""

from arco._arco import ArtificialPotentialField

__all__ = ["ArtificialPotentialField"]
