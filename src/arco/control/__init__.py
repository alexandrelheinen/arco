"""Control subpackage: feedback controllers, tracking, and object-centric control."""

from __future__ import annotations

from arco.control.actuator import ActuatorArray
from arco.control.avoidance import ArtificialPotentialField
from arco.control.base import Controller
from arco.control.joint_tracker import JointSpaceTracker
from arco.control.mpc import (
    DubinsPathFollowingMPC,
    DubinsVehicleLimits,
    JointSpaceMPC,
    JointSpaceMPCConfig,
    MPCController,
    MPCStepResult,
    MPCTracker,
    MPCTrackingLoop,
    PathFollowingMPCConfig,
    ReferencePath,
)
from arco.control.pid import PIDController
from arco.control.pure_pursuit import PurePursuitController
from arco.control.rigid_body import CircleBody, RigidBody, SquareBody
from arco.control.tracking import TrackingLoop

__all__ = [
    "ActuatorArray",
    "ArtificialPotentialField",
    "CircleBody",
    "Controller",
    "DubinsPathFollowingMPC",
    "DubinsVehicleLimits",
    "JointSpaceMPC",
    "JointSpaceMPCConfig",
    "JointSpaceTracker",
    "MPCController",
    "MPCStepResult",
    "MPCTracker",
    "MPCTrackingLoop",
    "PIDController",
    "PathFollowingMPCConfig",
    "PurePursuitController",
    "ReferencePath",
    "RigidBody",
    "SquareBody",
    "TrackingLoop",
]
