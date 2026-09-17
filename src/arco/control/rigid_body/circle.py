"""CircleBody, re-exported from the compiled extension.

The implementation is ``CircleBody`` in the ``arco-control`` crate,
registered back under its Python spelling by the binding layer and
reaching callers through :mod:`arco._arco`. It supplies the disk inertia
and the bounding radius that
:class:`~arco.control.rigid_body.base.RigidBody` leaves abstract.
"""

from arco._arco import CircleBody

__all__ = ["CircleBody"]
