"""ActuatorArray: N contact actuators around a 2-D rigid body."""

from __future__ import annotations

import math

import numpy as np

from arco.control.rigid_body.base import RigidBody


class ActuatorArray:
    """Array of N contact actuators around a 2-D rigid body.

    Each actuator is positioned at angle theta_i around the body boundary,
    at a standoff distance beyond the bounding radius.  It can apply a 2-D
    force decomposed into radial (normal) and tangential components.

    The grasp matrix G (3×2N) maps the actuator force vector
    f = [fr_1, ft_1, ..., fr_N, ft_N] to the body wrench W = [Fx, Fy, τ].

    Force allocation uses the Moore-Penrose pseudo-inverse G† so that
    W = G · G† · W_d = W_d (when W_d is in the column space of G).

    Args:
        actuator_count: Number of actuators N. Must be ≥ 3.
        standoff: Distance beyond bounding_radius at which actuators sit (m).

    Raises:
        ValueError: If actuator_count < 3.
    """

    def __init__(
        self,
        actuator_count: int = 4,
        standoff: float = 0.05,
    ) -> None:
        if actuator_count < 3:
            raise ValueError(
                f"actuator_count must be >= 3, got {actuator_count!r}."
            )
        self._actuator_count = actuator_count
        self._standoff = float(standoff)
        self._angles = np.linspace(
            0.0, 2.0 * math.pi, actuator_count, endpoint=False
        )

    @property
    def actuator_count(self) -> int:
        """Number of actuators."""
        return self._actuator_count

    @property
    def angles(self) -> np.ndarray:
        """Current actuator placement angles (radians), shape (N,)."""
        return self._angles.copy()

    def set_angles(self, angles: np.ndarray) -> None:
        """Override actuator placement angles.

        Args:
            angles: Array of N angles in radians, shape (N,).

        Raises:
            ValueError: If len(angles) != actuator_count.
        """
        angles = np.asarray(angles, dtype=float)
        if angles.shape != (self._actuator_count,):
            raise ValueError(
                f"Expected angles of shape ({self._actuator_count},), "
                f"got {angles.shape}."
            )
        self._angles = angles.copy()

    def grasp_matrix(self, body: RigidBody) -> np.ndarray:
        """Compute the 3×2N grasp matrix for the current actuator placement.

        Column 2i   corresponds to radial (normal-inward) force of actuator i.
        Column 2i+1 corresponds to tangential force of actuator i (CCW positive).

        For actuator i at angle theta_i on body with bounding radius R:
          contact point (body frame): r_i = R * (cos(theta_i), sin(theta_i))
          normal direction: n_i = (-cos(theta_i+psi), -sin(theta_i+psi))
          tangent direction: t_i = (-sin(theta_i+psi), cos(theta_i+psi))

          G[:, 2i]   = [n_ix, n_iy, r_i × n_i]
          G[:, 2i+1] = [t_ix, t_iy, r_i × t_i]

        where r_i × F = r_ix * F_iy - r_iy * F_ix (2D cross product).

        Args:
            body: The rigid body being grasped.

        Returns:
            Grasp matrix of shape (3, 2N).
        """
        n = self._actuator_count
        R = body.bounding_radius
        psi = float(body.pose[2])
        G = np.zeros((3, 2 * n), dtype=float)
        for i, theta in enumerate(self._angles):
            theta_w = theta + psi
            cos_w = math.cos(theta_w)
            sin_w = math.sin(theta_w)
            # Contact point in world frame relative to body center
            rx = R * cos_w
            ry = R * sin_w
            # Normal direction (inward)
            nx = -cos_w
            ny = -sin_w
            # Tangent direction (CCW)
            tx = -sin_w
            ty = cos_w
            # 2D cross product: r × F = rx*Fy - ry*Fx
            G[0, 2 * i] = nx
            G[1, 2 * i] = ny
            G[2, 2 * i] = rx * ny - ry * nx
            G[0, 2 * i + 1] = tx
            G[1, 2 * i + 1] = ty
            G[2, 2 * i + 1] = rx * ty - ry * tx
        return G

    def allocate_forces(
        self,
        desired_wrench: np.ndarray,
        body: RigidBody,
    ) -> np.ndarray:
        """Compute actuator forces to achieve the desired wrench.

        Uses the Moore-Penrose pseudo-inverse of the grasp matrix.

        Args:
            desired_wrench: [Fx, Fy, torque] in world frame.
            body: The rigid body being grasped.

        Returns:
            Force vector f of shape (2N,): [fr_1, ft_1, ..., fr_N, ft_N].
        """
        G = self.grasp_matrix(body)
        G_pinv = np.linalg.pinv(G)
        return G_pinv @ np.asarray(desired_wrench, dtype=float)

    def actuator_positions(self, body: RigidBody) -> np.ndarray:
        """Return world-frame positions of all actuators.

        Each actuator sits at standoff distance beyond the body's bounding
        radius.

        Args:
            body: The rigid body.

        Returns:
            Array of shape (N, 2) with (x, y) positions.
        """
        r = body.bounding_radius + self._standoff
        psi = float(body.pose[2])
        bx, by = float(body.pose[0]), float(body.pose[1])
        positions = np.empty((self._actuator_count, 2), dtype=float)
        for i, theta in enumerate(self._angles):
            theta_w = theta + psi
            positions[i, 0] = bx + r * math.cos(theta_w)
            positions[i, 1] = by + r * math.sin(theta_w)
        return positions

    def apply_to_body(
        self,
        forces: np.ndarray,
        body: RigidBody,
    ) -> None:
        """Apply the force vector to the body via apply_wrench().

        Converts the per-actuator force vector to individual wrenches
        and calls body.apply_wrench() for each actuator.

        Args:
            forces: Force vector of shape (2N,): [fr_1, ft_1, ...].
            body: Target rigid body.
        """
        G = self.grasp_matrix(body)
        wrench = G @ np.asarray(forces, dtype=float)
        body.apply_wrench(float(wrench[0]), float(wrench[1]), float(wrench[2]))

    def update_angles_for_target(
        self,
        body: RigidBody,
        target_wrench: np.ndarray,
    ) -> None:
        """Redistribute actuator angles to improve manipulability.

        Rotates all actuators collectively so that the primary force
        direction aligns with the net desired force direction.  This
        maximises the radial force contribution along the desired direction.

        Args:
            body: The rigid body.
            target_wrench: Desired [Fx, Fy, torque].
        """
        wrench = np.asarray(target_wrench, dtype=float)
        fx, fy = float(wrench[0]), float(wrench[1])
        psi = float(body.pose[2])
        # Desired inward direction in body frame
        desired_world = math.atan2(-fy, -fx) + math.pi
        desired_body = desired_world - psi
        n = self._actuator_count
        new_angles = np.empty(n, dtype=float)
        for i in range(n):
            new_angles[i] = desired_body + 2.0 * math.pi * i / n
        self._angles = new_angles
