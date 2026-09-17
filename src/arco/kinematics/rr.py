"""Planar RR arm kinematics, re-exported from the compiled extension.

The implementation is ``RrRobot`` in the ``arco-kinematics`` crate,
registered back under its Python spelling by the binding layer and
reaching callers through :mod:`arco._arco`.

The arm is mounted at the world origin and works in the XY plane. Joint 1
rotates the first link of length *l1* around the Z axis, and joint 2
rotates the second link of length *l2* relative to the first.

Example::

    robot = RRRobot(l1=1.0, l2=0.8)
    x, y = robot.forward_kinematics(0.0, 0.0)   # (1.8, 0.0)
    solutions = robot.inverse_kinematics(1.4, 0.5)
"""

from arco._arco import RRRobot

__all__ = ["RRRobot"]
