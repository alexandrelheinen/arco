"""Actuator array, re-exported from the compiled extension.

The implementation is ``ActuatorArray`` in the ``arco-control`` crate,
paired with ``ActuatorSettings`` and ``GraspMatrix``, registered back
under its Python spelling by the binding layer and reaching callers
through :mod:`arco._arco`. The array places N actuators around a
:class:`~arco.control.rigid_body.base.RigidBody`, builds the grasp matrix,
allocates a desired wrench across the contacts, and runs the second-order
placement dynamics.
"""

from arco._arco import ActuatorArray

__all__ = ["ActuatorArray"]
