"""Path-following MPC sub-package (CasADi optional)."""

from __future__ import annotations

from arco.control.mpc.base import MPCTracker
from arco.control.mpc.controller import MPCController
from arco.control.mpc.costs import forward_cone_factor, obstacle_barrier
from arco.control.mpc.joint_space import JointSpaceMPC, JointSpaceMPCConfig
from arco.control.mpc.path_following import (
    DubinsPathFollowingMPC,
    DubinsVehicleLimits,
    PathFollowingMPCConfig,
)
from arco.control.mpc.reference_path import ReferencePath
from arco.control.mpc.result import MPCStepResult
from arco.control.mpc.tracking_loop import MPCTrackingLoop

__all__ = [
    "DubinsPathFollowingMPC",
    "DubinsVehicleLimits",
    "JointSpaceMPC",
    "JointSpaceMPCConfig",
    "MPCController",
    "MPCStepResult",
    "MPCTracker",
    "MPCTrackingLoop",
    "PathFollowingMPCConfig",
    "ReferencePath",
    "forward_cone_factor",
    "obstacle_barrier",
]
