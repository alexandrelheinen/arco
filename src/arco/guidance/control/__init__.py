"""Control subpackage: feedback controllers and tracking loop.

.. deprecated::
    Import from :mod:`arco.control` instead of ``arco.guidance.control``.
    This module is kept for backward compatibility only.
"""

from arco.control.base import Controller
from arco.control.mpc import MPCController
from arco.control.pid import PIDController
from arco.control.pure_pursuit import PurePursuitController
from arco.control.tracking import TrackingLoop

__all__ = [
    "Controller",
    "MPCController",
    "PIDController",
    "PurePursuitController",
    "TrackingLoop",
]
