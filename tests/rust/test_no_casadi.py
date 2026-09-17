"""The compiled package never reaches for CasADi (FR-DEP-01, ADR-002).

The predictive controllers were the only callers, and they now build a
convex program the ``arco-control`` crate solves with Clarabel. A CasADi
import creeping back in would mean an optional heavy dependency had become
load-bearing again without anyone deciding so.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np

from arco.control.mpc import (
    DubinsPathFollowingMPC,
    DubinsVehicleLimits,
    JointSpaceMPC,
    PathFollowingMPCConfig,
)

_EXERCISE = """
import sys

import numpy as np

import arco
from arco.control.mpc import (
    DubinsPathFollowingMPC,
    DubinsVehicleLimits,
    JointSpaceMPC,
    PathFollowingMPCConfig,
)

tracker = DubinsPathFollowingMPC(
    vehicle_limits=DubinsVehicleLimits(1.0, 0.0, 1.5, 1.0, 2.0),
    config=PathFollowingMPCConfig(),
)
tracker.set_reference([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])
tracker.step((0.0, 0.1, 0.0), speed=0.2, turn_rate=0.0, dt=0.05)

joint = JointSpaceMPC(1.0, 2.0)
joint.reset(np.zeros(2))
joint.step(np.array([0.1, 0.0]), 0.05)

imported = [name for name in sys.modules if name.split(".")[0] == "casadi"]
print(",".join(imported))
"""


def test_a_full_mpc_step_imports_no_casadi_module() -> None:
    """Driving both controllers leaves ``sys.modules`` free of CasADi."""
    finished = subprocess.run(
        [sys.executable, "-c", _EXERCISE],
        capture_output=True,
        check=True,
        text=True,
    )
    assert finished.stdout.strip() == ""


def test_the_controllers_run_without_the_optional_dependency() -> None:
    """Both controllers solve a step in this interpreter, CasADi or not."""
    tracker = DubinsPathFollowingMPC(
        vehicle_limits=DubinsVehicleLimits(1.0, 0.0, 1.5, 1.0, 2.0),
        config=PathFollowingMPCConfig(),
    )
    tracker.set_reference([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])
    result = tracker.step((0.0, 0.1, 0.0), speed=0.2, turn_rate=0.0, dt=0.05)
    assert result.solver_success

    joint = JointSpaceMPC(1.0, 2.0)
    joint.reset(np.zeros(2))
    moved = joint.step(np.array([0.1, 0.0]), 0.05)
    assert moved.shape == (2,)
