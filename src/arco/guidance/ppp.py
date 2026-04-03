"""PPPRobot: triple prismatic joint robot with independent 3-axis control.

A PPP robot (triple prismatic joint) is a Cartesian robot with three
independently controlled linear actuators, commonly seen in 3D printers,
CNC machines, and gantry systems. Each axis has independent acceleration
limits and velocity saturation.
"""

from __future__ import annotations

import numpy as np


class PPPRobot:
    """Cartesian robot with three independently controlled prismatic joints.

    Models a PPP (Prismatic-Prismatic-Prismatic) configuration robot with
    independent acceleration control along x, y, and z axes. Each axis has
    independent velocity and acceleration limits, similar to a 3D printer
    or CNC machine.

    The robot operates within a box-shaped workspace defined by
    ``work_volume``. Motion outside this volume is constrained.

    Attributes:
        x: Current x position (metres).
        y: Current y position (metres).
        z: Current z position (metres).
        work_volume: Workspace bounds as ``[[x_min, x_max], [y_min, y_max],
            [z_min, z_max]]`` (metres).
        max_velocity: Maximum velocity along each axis (m/s).
        max_acceleration: Maximum acceleration along each axis (m/s²).
    """

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        work_volume: list[list[float]] | None = None,
        max_velocity: float = 1.0,
        max_acceleration: float = 0.5,
    ) -> None:
        """Initialize PPPRobot.

        Args:
            x: Initial x position in world frame (metres).
            y: Initial y position in world frame (metres).
            z: Initial z position in world frame (metres).
            work_volume: Workspace bounds as ``[[x_min, x_max],
                [y_min, y_max], [z_min, z_max]]``. Defaults to
                ``[[0, 10], [0, 10], [0, 10]]`` if not specified.
            max_velocity: Maximum velocity along each axis (m/s).
            max_acceleration: Maximum acceleration along each axis (m/s²).
        """
        self.x = x
        self.y = y
        self.z = z

        if work_volume is None:
            self.work_volume = [[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]]
        else:
            self.work_volume = work_volume

        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration

        self._vx: float = 0.0
        self._vy: float = 0.0
        self._vz: float = 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def position(self) -> np.ndarray:
        """Current position as ``(x, y, z)`` numpy array."""
        return np.array([self.x, self.y, self.z], dtype=float)

    @property
    def velocity(self) -> np.ndarray:
        """Current velocity as ``(vx, vy, vz)`` numpy array."""
        return np.array([self._vx, self._vy, self._vz], dtype=float)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        """Reset robot state to a new position with zero velocity.

        Args:
            x: New x position (metres).
            y: New y position (metres).
            z: New z position (metres).
        """
        self.x = x
        self.y = y
        self.z = z
        self._vx = 0.0
        self._vy = 0.0
        self._vz = 0.0

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def step(self, ax: float, ay: float, az: float, dt: float) -> np.ndarray:
        """Integrate dynamics one step with acceleration commands.

        Applies acceleration to each axis independently, saturates velocities
        to ``max_velocity``, and integrates position using forward-Euler.
        Constrains the resulting position to the workspace volume.

        Args:
            ax: Desired acceleration along x axis (m/s²).
            ay: Desired acceleration along y axis (m/s²).
            az: Desired acceleration along z axis (m/s²).
            dt: Time step duration (s).

        Returns:
            Updated position as ``(x, y, z)`` numpy array.
        """
        # Clip accelerations to max limits
        ax = float(np.clip(ax, -self.max_acceleration, self.max_acceleration))
        ay = float(np.clip(ay, -self.max_acceleration, self.max_acceleration))
        az = float(np.clip(az, -self.max_acceleration, self.max_acceleration))

        # Integrate velocities
        self._vx = float(
            np.clip(self._vx + ax * dt, -self.max_velocity, self.max_velocity)
        )
        self._vy = float(
            np.clip(self._vy + ay * dt, -self.max_velocity, self.max_velocity)
        )
        self._vz = float(
            np.clip(self._vz + az * dt, -self.max_velocity, self.max_velocity)
        )

        # Integrate positions
        self.x += self._vx * dt
        self.y += self._vy * dt
        self.z += self._vz * dt

        # Constrain to workspace
        self.x = float(
            np.clip(self.x, self.work_volume[0][0], self.work_volume[0][1])
        )
        self.y = float(
            np.clip(self.y, self.work_volume[1][0], self.work_volume[1][1])
        )
        self.z = float(
            np.clip(self.z, self.work_volume[2][0], self.work_volume[2][1])
        )

        # Stop velocity when hitting workspace boundary
        if (
            self.x <= self.work_volume[0][0]
            or self.x >= self.work_volume[0][1]
        ):
            self._vx = 0.0
        if (
            self.y <= self.work_volume[1][0]
            or self.y >= self.work_volume[1][1]
        ):
            self._vy = 0.0
        if (
            self.z <= self.work_volume[2][0]
            or self.z >= self.work_volume[2][1]
        ):
            self._vz = 0.0

        return self.position
