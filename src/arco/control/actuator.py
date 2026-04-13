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

    Each actuator axis (angular θ_i and radial r_i) is modelled as a
    second-order closed-loop system driven toward a reference setpoint::

        ξ̈_i = −4 ζ Ω ξ̇_i − 2 Ω² (ξ_i − ξ_i*)

    which results from combining the plant
    ``ξ̈ + 2ζΩξ̇ + Ω²ξ = Ω²ξ* + u`` with PD gains
    ``k_p = Ω²``, ``k_d = 2ζΩ``.

    Contact forces are produced by a spring at the radial axis:
    ``F_i = k_s · max(0, r_nom_i − r_i)`` where
    ``r_nom_i = bounding_radius + standoff``.

    Args:
        actuator_count: Number of actuators N. Must be ≥ 3.
        standoff: Distance beyond bounding_radius at which actuators sit (m).
        omega: Natural frequency Ω (rad/s) of the actuator second-order loop.
        zeta: Damping ratio ζ of the actuator second-order loop.
        spring_stiffness: Spring stiffness k_s (N/m) for contact force.

    Raises:
        ValueError: If actuator_count < 3.
    """

    def __init__(
        self,
        actuator_count: int = 4,
        standoff: float = 0.05,
        omega: float = 10.0,
        zeta: float = 0.7,
        spring_stiffness: float = 100.0,
    ) -> None:
        if actuator_count < 3:
            raise ValueError(
                f"actuator_count must be >= 3, got {actuator_count!r}."
            )
        self._actuator_count = actuator_count
        self._standoff = float(standoff)
        self._omega = float(omega)
        self._zeta = float(zeta)
        self._spring_stiffness = float(spring_stiffness)
        self._angles = np.linspace(
            0.0, 2.0 * math.pi, actuator_count, endpoint=False
        )
        self._angle_velocities = np.zeros(actuator_count, dtype=float)
        self._ref_angles = self._angles.copy()
        # Radial state is initialised lazily via init_radii(body).
        self._radii: np.ndarray | None = None
        self._radii_velocities: np.ndarray | None = None
        self._ref_radii: np.ndarray | None = None

    @property
    def actuator_count(self) -> int:
        """Number of actuators."""
        return self._actuator_count

    @property
    def omega(self) -> float:
        """Natural frequency Ω (rad/s) of the actuator second-order loop."""
        return self._omega

    @property
    def zeta(self) -> float:
        """Damping ratio ζ of the actuator second-order loop."""
        return self._zeta

    @property
    def spring_stiffness(self) -> float:
        """Spring stiffness k_s (N/m) used for the contact force model."""
        return self._spring_stiffness

    @property
    def angles(self) -> np.ndarray:
        """Current actuator placement angles (radians), shape (N,)."""
        return self._angles.copy()

    @property
    def angle_velocities(self) -> np.ndarray:
        """Current angular velocities θ̇_i (rad/s), shape (N,)."""
        return self._angle_velocities.copy()

    @property
    def ref_angles(self) -> np.ndarray:
        """Reference (setpoint) angles θ_i* (radians), shape (N,)."""
        return self._ref_angles.copy()

    @property
    def radii(self) -> np.ndarray | None:
        """Current radial positions r_i (m), shape (N,), or ``None``."""
        return None if self._radii is None else self._radii.copy()

    @property
    def radii_velocities(self) -> np.ndarray | None:
        """Current radial velocities ṙ_i (m/s), shape (N,), or ``None``."""
        return (
            None
            if self._radii_velocities is None
            else self._radii_velocities.copy()
        )

    @property
    def ref_radii(self) -> np.ndarray | None:
        """Reference radial positions r_i* (m), shape (N,), or ``None``."""
        return None if self._ref_radii is None else self._ref_radii.copy()

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

    def allocate_radial_forces(
        self,
        desired_wrench: np.ndarray,
        body: RigidBody,
    ) -> np.ndarray:
        """Compute radial-only actuator forces to best achieve the desired wrench.

        Uses the pseudo-inverse of the radial sub-matrix of the grasp matrix
        (columns 0, 2, 4, … of G).  Tangential forces are set to zero.

        This allocation is used with the spring-based contact force model,
        where only the radial (normal) axis produces force.

        Args:
            desired_wrench: [Fx, Fy, torque] in world frame.
            body: The rigid body being grasped.

        Returns:
            Force vector f of shape (2N,): [fr_1, 0, fr_2, 0, ..., fr_N, 0].
        """
        G = self.grasp_matrix(body)
        G_radial = G[:, 0::2]  # shape (3, N)
        G_radial_pinv = np.linalg.pinv(G_radial)
        fr = G_radial_pinv @ np.asarray(desired_wrench, dtype=float)
        forces = np.zeros(2 * self._actuator_count, dtype=float)
        forces[0::2] = fr
        return forces

    def actuator_positions(self, body: RigidBody) -> np.ndarray:
        """Return world-frame positions of all actuators.

        When radial state has been initialised (via :meth:`init_radii` or
        :meth:`step_actuators`), the actual per-actuator radius ``r_i`` is
        used.  Otherwise the nominal standoff distance is used.

        Args:
            body: The rigid body.

        Returns:
            Array of shape (N, 2) with (x, y) positions.
        """
        r_nom = body.bounding_radius + self._standoff
        psi = float(body.pose[2])
        bx, by = float(body.pose[0]), float(body.pose[1])
        positions = np.empty((self._actuator_count, 2), dtype=float)
        for i, theta in enumerate(self._angles):
            theta_w = theta + psi
            r = float(self._radii[i]) if self._radii is not None else r_nom
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
        """Set reference angles to align actuators with the desired wrench.

        Rotates all actuators collectively so that the primary force
        direction aligns with the net desired force direction.  This
        maximises the radial force contribution along the desired direction.

        The computed angles become the setpoints ``θ_i*`` fed to the
        second-order angular loop.  Actual angles evolve toward these
        references via :meth:`step_actuators`.

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
        self._ref_angles = desired_body + 2.0 * math.pi * np.arange(n) / n

    # ------------------------------------------------------------------
    # Second-order actuator dynamics
    # ------------------------------------------------------------------

    def init_radii(self, body: RigidBody) -> None:
        """Initialise radial state from the current body geometry.

        Sets each actuator's radial position to the nominal contact
        distance ``bounding_radius + standoff`` and zero velocity.
        Must be called before :meth:`step_actuators` when the spring-based
        force model is used.

        Args:
            body: The rigid body being manipulated.
        """
        r0 = body.bounding_radius + self._standoff
        self._radii = np.full(self._actuator_count, r0, dtype=float)
        self._radii_velocities = np.zeros(self._actuator_count, dtype=float)
        self._ref_radii = self._radii.copy()

    def compute_ref_radii(
        self,
        body: RigidBody,
        desired_forces: np.ndarray,
    ) -> None:
        """Compute and store reference radii via the spring inversion law.

        For each actuator the reference radial position is chosen so that,
        when the actuator converges, the spring force equals the desired
        radial force::

            r_i* = r_nom_i − F̃_i / k_s

        where ``F̃_i = F_i* + F_bias`` is the biased desired force and
        ``r_nom_i = bounding_radius + standoff`` is evaluated at the
        **actual** current angle θ_i (Step 4 of the OCC loop).

        A symmetric precompression bias ``F_bias = max(0, −min_i F_i*)`` is
        added so that all reference compressions are non-negative.  For an
        array with evenly-spaced actuators this bias cancels in the net
        wrench, so the body-level control is unaffected.

        If radial state has not been initialised, :meth:`init_radii` is
        called automatically.

        Args:
            body: The rigid body (provides bounding_radius).
            desired_forces: Force vector of shape (2N,):
                ``[fr_1, ft_1, ..., fr_N, ft_N]``.  The radial components
                ``fr_i = desired_forces[2*i]`` are used for the inversion.
        """
        if self._radii is None:
            self.init_radii(body)
        r_nom = body.bounding_radius + self._standoff
        f = np.asarray(desired_forces, dtype=float)
        fr_values = np.array(
            [float(f[2 * i]) for i in range(self._actuator_count)]
        )
        # Precompression bias so that all F̃_i ≥ 0 (springs can only push).
        bias = max(0.0, -float(np.min(fr_values)))
        biased_fr = fr_values + bias
        ref = r_nom - biased_fr / self._spring_stiffness
        self._ref_radii = ref

    def step_actuators(self, dt: float) -> None:
        """Integrate the second-order actuator dynamics by one time step.

        Both the angular and radial axes obey the closed-loop ODE::

            ξ̈ = −4 ζ Ω ξ̇ − 2 Ω² (ξ − ξ*)

        which corresponds to a second-order plant plus PD gains
        ``k_p = Ω²``, ``k_d = 2ζΩ``.

        If radial state has not been initialised this method only integrates
        the angular dynamics.

        Args:
            dt: Integration time step in seconds.
        """
        omega = self._omega
        zeta = self._zeta
        a_coeff = 4.0 * zeta * omega
        b_coeff = 2.0 * omega * omega

        # Angular dynamics
        err_angle = self._angles - self._ref_angles
        ang_acc = -a_coeff * self._angle_velocities - b_coeff * err_angle
        self._angle_velocities += ang_acc * dt
        self._angles += self._angle_velocities * dt

        # Radial dynamics (if initialised)
        if self._radii is not None and self._ref_radii is not None:
            err_radii = self._radii - self._ref_radii
            rad_acc = -a_coeff * self._radii_velocities - b_coeff * err_radii
            self._radii_velocities += rad_acc * dt
            self._radii += self._radii_velocities * dt

    def spring_forces(self, body: RigidBody) -> np.ndarray:
        """Compute the actuator force vector from the spring contact model.

        The radial (normal) force for actuator *i* is::

            fr_i = k_s · max(0, r_nom_i − r_i)

        and the tangential force is zero (pure normal contact).

        If radial state has not been initialised, :meth:`init_radii` is
        called automatically and the returned forces are zero.

        Args:
            body: The rigid body (provides bounding_radius).

        Returns:
            Force vector of shape (2N,): ``[fr_1, 0, ..., fr_N, 0]``.
        """
        if self._radii is None:
            self.init_radii(body)
        assert self._radii is not None
        r_nom = body.bounding_radius + self._standoff
        forces = np.zeros(2 * self._actuator_count, dtype=float)
        for i in range(self._actuator_count):
            compression = r_nom - float(self._radii[i])
            forces[2 * i] = self._spring_stiffness * max(0.0, compression)
        return forces

    def apply_spring_forces_to_body(self, body: RigidBody) -> None:
        """Compute spring forces and apply them to the body.

        Convenience wrapper that calls :meth:`spring_forces` and then
        applies the result through the grasp matrix via
        :meth:`apply_to_body`.

        Args:
            body: Target rigid body.
        """
        forces = self.spring_forces(body)
        self.apply_to_body(forces, body)
