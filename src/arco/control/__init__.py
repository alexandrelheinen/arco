"""Control subpackage: feedback controllers, tracking, and object-centric control."""

from __future__ import annotations

from arco.control.actuator import ActuatorArray
from arco.control.base import Controller
from arco.control.mpc import MPCController
from arco.control.pid import PIDController
from arco.control.pure_pursuit import PurePursuitController
from arco.control.rigid_body import CircleBody, RigidBody, SquareBody
from arco.control.tracking import TrackingLoop

__all__ = [
    "ActuatorArray",
    "CircleBody",
    "Controller",
    "MPCController",
    "PIDController",
    "PurePursuitController",
    "RigidBody",
    "SquareBody",
    "TrackingLoop",
]
