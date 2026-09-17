"""SquareBody, re-exported from the compiled extension.

The implementation is ``SquareBody`` in the ``arco-control`` crate,
registered back under its Python spelling by the binding layer and
reaching callers through :mod:`arco._arco`. It supplies the square-plate
inertia and the bounding radius that
:class:`~arco.control.rigid_body.base.RigidBody` leaves abstract, plus the
``corners`` of the footprint in world coordinates.
"""

from arco._arco import SquareBody

__all__ = ["SquareBody"]
