"""Planar RRP arm kinematics, re-exported from the compiled extension.

The implementation is ``RrpRobot`` in the ``arco-kinematics`` crate,
registered back under its Python spelling by the binding layer and
reaching callers through :mod:`arco._arco`.

The arm adds a vertical prismatic joint to the planar RR arm, giving the
SCARA-like layout: the two revolute joints place the end effector in XY
and the prismatic joint sets Z between *z_min* and *z_max*.

Example::

    robot = RRPRobot(l1=1.0, l2=0.8, z_min=0.0, z_max=4.0)
    x, y, z = robot.forward_kinematics(0.0, 0.0, 1.5)
    solutions = robot.inverse_kinematics_xy(1.4, 0.5)
"""

from arco._arco import RRPRobot

__all__ = ["RRPRobot"]
