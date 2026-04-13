"""Tests for arco.control.actuator — ActuatorArray."""

from __future__ import annotations

import math

import numpy as np
import pytest

from arco.control.actuator import ActuatorArray
from arco.control.rigid_body import CircleBody, SquareBody

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def square() -> SquareBody:
    return SquareBody(mass=5.0, side_length=0.5)


@pytest.fixture()
def circle() -> CircleBody:
    return CircleBody(mass=5.0, radius=0.3)


@pytest.fixture()
def array4() -> ActuatorArray:
    return ActuatorArray(actuator_count=4, standoff=0.05)


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestActuatorArrayInit:
    def test_default_count(self) -> None:
        a = ActuatorArray()
        assert a.actuator_count == 4

    def test_custom_count(self) -> None:
        a = ActuatorArray(actuator_count=6)
        assert a.actuator_count == 6

    def test_angles_length(self, array4: ActuatorArray) -> None:
        assert len(array4.angles) == 4

    def test_count_less_than_3_raises(self) -> None:
        with pytest.raises(ValueError, match="actuator_count"):
            ActuatorArray(actuator_count=2)

    def test_count_exactly_3_ok(self) -> None:
        a = ActuatorArray(actuator_count=3)
        assert a.actuator_count == 3

    def test_angles_evenly_spaced(self, array4: ActuatorArray) -> None:
        angles = array4.angles
        diffs = np.diff(angles)
        expected_step = 2.0 * math.pi / 4
        np.testing.assert_array_almost_equal(diffs, [expected_step] * 3)


class TestSetAngles:
    def test_set_valid_angles(self, array4: ActuatorArray) -> None:
        new_angles = np.array([0.0, 1.0, 2.0, 3.0])
        array4.set_angles(new_angles)
        np.testing.assert_array_almost_equal(array4.angles, new_angles)

    def test_set_wrong_length_raises(self, array4: ActuatorArray) -> None:
        with pytest.raises(ValueError):
            array4.set_angles(np.array([0.0, 1.0, 2.0]))


# ---------------------------------------------------------------------------
# Grasp matrix
# ---------------------------------------------------------------------------


class TestGraspMatrix:
    def test_shape(self, array4: ActuatorArray, square: SquareBody) -> None:
        G = array4.grasp_matrix(square)
        assert G.shape == (3, 8)

    def test_shape_circle(
        self, array4: ActuatorArray, circle: CircleBody
    ) -> None:
        G = array4.grasp_matrix(circle)
        assert G.shape == (3, 8)

    def test_shape_6_actuators(self, square: SquareBody) -> None:
        a = ActuatorArray(actuator_count=6)
        G = a.grasp_matrix(square)
        assert G.shape == (3, 12)

    def test_rank_sufficient(
        self, array4: ActuatorArray, circle: CircleBody
    ) -> None:
        G = array4.grasp_matrix(circle)
        # With 4 actuators (8 cols), rank should be 3
        assert np.linalg.matrix_rank(G) == 3


# ---------------------------------------------------------------------------
# Force allocation
# ---------------------------------------------------------------------------


class TestForceAllocation:
    def test_achieve_desired_wrench_circle(
        self, array4: ActuatorArray, circle: CircleBody
    ) -> None:
        w_desired = np.array([2.0, -1.0, 0.5])
        f = array4.allocate_forces(w_desired, circle)
        G = array4.grasp_matrix(circle)
        w_actual = G @ f
        np.testing.assert_array_almost_equal(w_actual, w_desired, decimal=10)

    def test_achieve_desired_wrench_square(
        self, array4: ActuatorArray, square: SquareBody
    ) -> None:
        w_desired = np.array([1.0, 0.0, 0.0])
        f = array4.allocate_forces(w_desired, square)
        G = array4.grasp_matrix(square)
        w_actual = G @ f
        np.testing.assert_array_almost_equal(w_actual, w_desired, decimal=10)

    def test_zero_wrench_gives_zero_forces(
        self, array4: ActuatorArray, circle: CircleBody
    ) -> None:
        f = array4.allocate_forces(np.zeros(3), circle)
        np.testing.assert_array_almost_equal(f, np.zeros(8), decimal=10)

    def test_output_shape(
        self, array4: ActuatorArray, circle: CircleBody
    ) -> None:
        f = array4.allocate_forces(np.array([1.0, 0.0, 0.0]), circle)
        assert f.shape == (8,)


# ---------------------------------------------------------------------------
# Actuator positions
# ---------------------------------------------------------------------------


class TestActuatorPositions:
    def test_shape(self, array4: ActuatorArray, circle: CircleBody) -> None:
        positions = array4.actuator_positions(circle)
        assert positions.shape == (4, 2)

    def test_distance_from_body(
        self, array4: ActuatorArray, circle: CircleBody
    ) -> None:
        positions = array4.actuator_positions(circle)
        expected_dist = circle.bounding_radius + 0.05
        for pos in positions:
            dist = math.hypot(pos[0], pos[1])
            assert dist == pytest.approx(expected_dist, rel=1e-6)

    def test_positions_shift_with_body_translation(
        self, array4: ActuatorArray, circle: CircleBody
    ) -> None:
        p0 = array4.actuator_positions(circle)
        circle.reset(x=1.0, y=2.0)
        p1 = array4.actuator_positions(circle)
        diff = p1 - p0
        np.testing.assert_array_almost_equal(diff, np.full((4, 2), [1.0, 2.0]))


# ---------------------------------------------------------------------------
# Apply to body
# ---------------------------------------------------------------------------


class TestApplyToBody:
    def test_apply_moves_body(
        self, array4: ActuatorArray, circle: CircleBody
    ) -> None:
        w_desired = np.array([10.0, 0.0, 0.0])
        forces = array4.allocate_forces(w_desired, circle)
        array4.apply_to_body(forces, circle)
        circle.step(dt=1.0)
        # Should have moved in x direction
        assert circle.pose[0] > 0.0

    def test_apply_zero_forces_no_motion(
        self, array4: ActuatorArray, circle: CircleBody
    ) -> None:
        array4.apply_to_body(np.zeros(8), circle)
        circle.step(dt=1.0)
        np.testing.assert_array_almost_equal(circle.pose, [0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Second-order dynamics parameters
# ---------------------------------------------------------------------------


class TestSecondOrderParams:
    def test_default_omega(self) -> None:
        a = ActuatorArray()
        assert a.omega == pytest.approx(10.0)

    def test_default_zeta(self) -> None:
        a = ActuatorArray()
        assert a.zeta == pytest.approx(0.7)

    def test_default_spring_stiffness(self) -> None:
        a = ActuatorArray()
        assert a.spring_stiffness == pytest.approx(100.0)

    def test_custom_params(self) -> None:
        a = ActuatorArray(omega=5.0, zeta=1.0, spring_stiffness=200.0)
        assert a.omega == pytest.approx(5.0)
        assert a.zeta == pytest.approx(1.0)
        assert a.spring_stiffness == pytest.approx(200.0)

    def test_angle_velocities_initially_zero(
        self, array4: ActuatorArray
    ) -> None:
        np.testing.assert_array_almost_equal(
            array4.angle_velocities, np.zeros(4)
        )

    def test_radii_initially_none(self, array4: ActuatorArray) -> None:
        assert array4.radii is None

    def test_ref_angles_match_initial_angles(
        self, array4: ActuatorArray
    ) -> None:
        np.testing.assert_array_almost_equal(array4.ref_angles, array4.angles)


# ---------------------------------------------------------------------------
# init_radii
# ---------------------------------------------------------------------------


class TestInitRadii:
    def test_radii_set_to_nominal(
        self, array4: ActuatorArray, circle: CircleBody
    ) -> None:
        array4.init_radii(circle)
        expected = circle.bounding_radius + 0.05
        assert array4.radii is not None
        np.testing.assert_array_almost_equal(
            array4.radii, np.full(4, expected)
        )

    def test_radii_velocities_zero_after_init(
        self, array4: ActuatorArray, circle: CircleBody
    ) -> None:
        array4.init_radii(circle)
        assert array4.radii_velocities is not None
        np.testing.assert_array_almost_equal(
            array4.radii_velocities, np.zeros(4)
        )

    def test_ref_radii_set_to_nominal(
        self, array4: ActuatorArray, circle: CircleBody
    ) -> None:
        array4.init_radii(circle)
        expected = circle.bounding_radius + 0.05
        assert array4.ref_radii is not None
        np.testing.assert_array_almost_equal(
            array4.ref_radii, np.full(4, expected)
        )


# ---------------------------------------------------------------------------
# step_actuators — angular second-order dynamics
# ---------------------------------------------------------------------------


class TestStepActuators:
    def test_angles_converge_to_ref(self, circle: CircleBody) -> None:
        """Angles must converge to reference under second-order dynamics."""
        a = ActuatorArray(actuator_count=4, omega=20.0, zeta=0.7)
        a.init_radii(circle)
        ref = np.array([0.5, 1.5, 2.5, 3.5])
        a._ref_angles = ref.copy()
        dt = 0.01
        for _ in range(500):
            a.step_actuators(dt)
        np.testing.assert_array_almost_equal(a.angles, ref, decimal=2)

    def test_angle_velocity_nonzero_during_transient(
        self, circle: CircleBody
    ) -> None:
        a = ActuatorArray(actuator_count=4, omega=5.0, zeta=0.7)
        a.init_radii(circle)
        a._ref_angles = np.zeros(4)
        a._angles = np.ones(4) * 0.5
        a.step_actuators(0.01)
        # Angular velocity should be nonzero immediately
        assert not np.allclose(a.angle_velocities, 0.0)

    def test_radii_converge_to_ref(self, circle: CircleBody) -> None:
        """Radial positions must converge to reference."""
        a = ActuatorArray(actuator_count=4, omega=20.0, zeta=0.7)
        a.init_radii(circle)
        r_nom = circle.bounding_radius + 0.05
        # Push reference slightly inward (simulate desired contact force)
        a._ref_radii = np.full(4, r_nom - 0.02)
        dt = 0.01
        for _ in range(500):
            a.step_actuators(dt)
        np.testing.assert_array_almost_equal(
            a.radii, np.full(4, r_nom - 0.02), decimal=2
        )

    def test_no_radii_init_skips_radial_dynamics(
        self, array4: ActuatorArray
    ) -> None:
        # Calling step_actuators without init_radii should not raise
        array4.step_actuators(0.01)
        assert array4.radii is None


# ---------------------------------------------------------------------------
# compute_ref_radii — spring inversion (Step 4)
# ---------------------------------------------------------------------------


class TestComputeRefRadii:
    def test_ref_radii_computed_from_desired_forces(
        self, array4: ActuatorArray, circle: CircleBody
    ) -> None:
        """r_i* = r_nom - (fr_i + bias) / k_s (spring inversion with bias)."""
        array4.init_radii(circle)
        k = array4.spring_stiffness
        r_nom = circle.bounding_radius + 0.05
        # All-positive desired radial forces → bias = 0
        fr = np.array([10.0, 5.0, 8.0, 3.0])
        f = np.zeros(8)
        for i, fri in enumerate(fr):
            f[2 * i] = fri
        array4.compute_ref_radii(circle, f)
        expected = r_nom - fr / k  # bias = 0 since all fr > 0
        assert array4.ref_radii is not None
        np.testing.assert_array_almost_equal(array4.ref_radii, expected)

    def test_auto_init_if_not_initialised(
        self, array4: ActuatorArray, circle: CircleBody
    ) -> None:
        # Should not raise even without calling init_radii first
        f = np.zeros(8)
        array4.compute_ref_radii(circle, f)
        assert array4.ref_radii is not None

    def test_negative_desired_force_adds_bias(
        self, array4: ActuatorArray, circle: CircleBody
    ) -> None:
        """Negative desired forces trigger precompression bias."""
        array4.init_radii(circle)
        k = array4.spring_stiffness
        r_nom = circle.bounding_radius + 0.05
        # One negative radial force (unphysical tension)
        fr = np.array([-2.0, 1.0, 3.0, 1.0])
        f = np.zeros(8)
        for i, fri in enumerate(fr):
            f[2 * i] = fri
        array4.compute_ref_radii(circle, f)
        # bias = 2.0 (to make -2 → 0)
        biased_fr = fr + 2.0
        expected = r_nom - biased_fr / k
        assert array4.ref_radii is not None
        np.testing.assert_array_almost_equal(array4.ref_radii, expected)


# ---------------------------------------------------------------------------
# spring_forces
# ---------------------------------------------------------------------------


class TestSpringForces:
    def test_zero_compression_gives_zero_force(
        self, array4: ActuatorArray, circle: CircleBody
    ) -> None:
        """At nominal (r_i = r_nom) spring force is zero."""
        array4.init_radii(circle)
        forces = array4.spring_forces(circle)
        np.testing.assert_array_almost_equal(forces, np.zeros(8))

    def test_compression_gives_positive_radial_force(
        self, array4: ActuatorArray, circle: CircleBody
    ) -> None:
        array4.init_radii(circle)
        r_nom = circle.bounding_radius + 0.05
        # Move actuator inside nominal → compression
        array4._radii = np.full(4, r_nom - 0.01)
        forces = array4.spring_forces(circle)
        k = array4.spring_stiffness
        for i in range(4):
            assert forces[2 * i] == pytest.approx(k * 0.01, rel=1e-6)

    def test_tension_gives_zero_force(
        self, array4: ActuatorArray, circle: CircleBody
    ) -> None:
        """Actuators cannot pull; tension (r > r_nom) gives zero force."""
        array4.init_radii(circle)
        r_nom = circle.bounding_radius + 0.05
        array4._radii = np.full(4, r_nom + 0.05)
        forces = array4.spring_forces(circle)
        for i in range(4):
            assert forces[2 * i] == 0.0

    def test_tangential_forces_always_zero(
        self, array4: ActuatorArray, circle: CircleBody
    ) -> None:
        array4.init_radii(circle)
        forces = array4.spring_forces(circle)
        for i in range(4):
            assert forces[2 * i + 1] == 0.0


# ---------------------------------------------------------------------------
# update_angles_for_target — sets reference angles (not actual)
# ---------------------------------------------------------------------------


class TestUpdateAnglesForTarget:
    def test_sets_ref_angles_not_actual(
        self, array4: ActuatorArray, circle: CircleBody
    ) -> None:
        """Actual angles stay unchanged; only ref_angles are updated."""
        original_angles = array4.angles.copy()
        # Use a body with non-zero pose so that the desired angle differs.
        body = CircleBody(mass=5.0, radius=0.3)
        body.reset(x=1.0, y=0.0, psi=math.pi / 4)
        w = np.array([1.0, 0.0, 0.0])
        array4.update_angles_for_target(body, w)
        # Actual angles unchanged; reference angles updated
        np.testing.assert_array_almost_equal(array4.angles, original_angles)
        # ref_angles should differ because psi ≠ 0 shifts desired_body
        assert not np.allclose(array4.ref_angles, original_angles)

    def test_ref_angles_change_with_wrench(
        self, array4: ActuatorArray, circle: CircleBody
    ) -> None:
        """Different wrenches produce different reference angle sets."""
        array4.update_angles_for_target(circle, np.array([1.0, 0.0, 0.0]))
        ref_x = array4.ref_angles.copy()
        array4.update_angles_for_target(circle, np.array([0.0, 1.0, 0.0]))
        ref_y = array4.ref_angles.copy()
        assert not np.allclose(ref_x, ref_y)


# ---------------------------------------------------------------------------
# Convergence degradation test (acceptance criterion 5)
# ---------------------------------------------------------------------------


class TestConvergenceDegradation:
    """Show that slower actuators produce larger wrench residuals.

    A body is held at the origin; the desired wrench is a pure Fx force.
    With high Ω the actuators reach setpoints quickly → applied wrench
    ≈ desired wrench.  With low Ω the actuators lag → residual is larger,
    but the system still converges given enough time.
    """

    def _simulate_wrench_residual(
        self,
        omega: float,
        steps: int = 300,
        dt: float = 0.01,
    ) -> float:
        """Return the mean wrench residual averaged over the last 50 steps."""
        body = CircleBody(mass=5.0, radius=0.3)
        a = ActuatorArray(
            actuator_count=4,
            standoff=0.05,
            omega=omega,
            zeta=0.7,
            spring_stiffness=200.0,
        )
        a.init_radii(body)
        w_desired = np.array([5.0, 0.0, 0.0])
        residuals = []
        for _ in range(steps):
            a.update_angles_for_target(body, w_desired)
            f_des = a.allocate_radial_forces(w_desired, body)
            a.compute_ref_radii(body, f_des)
            a.step_actuators(dt)
            forces = a.spring_forces(body)
            G = a.grasp_matrix(body)
            w_actual = G @ forces
            residuals.append(float(np.linalg.norm(w_desired - w_actual)))
        return float(np.mean(residuals[-50:]))

    def test_high_omega_lower_residual_than_low_omega(self) -> None:
        """Fast actuators must produce smaller wrench residuals."""
        residual_fast = self._simulate_wrench_residual(omega=30.0)
        residual_slow = self._simulate_wrench_residual(omega=2.0)
        assert residual_fast < residual_slow, (
            f"Expected fast (Ω=30) residual {residual_fast:.4f} < "
            f"slow (Ω=2) residual {residual_slow:.4f}"
        )

    def test_fast_actuators_converge(self) -> None:
        """Fast actuators must produce small steady-state wrench residual."""
        residual = self._simulate_wrench_residual(omega=30.0, steps=600)
        assert (
            residual < 1.0
        ), f"Expected residual < 1.0 for fast actuators, got {residual:.4f}"

    def test_slow_actuators_still_reduce_residual_over_time(self) -> None:
        """Even slow actuators must improve from their initial residual."""
        body = CircleBody(mass=5.0, radius=0.3)
        a = ActuatorArray(
            actuator_count=4,
            standoff=0.05,
            omega=1.0,
            zeta=0.7,
            spring_stiffness=200.0,
        )
        a.init_radii(body)
        w_desired = np.array([5.0, 0.0, 0.0])
        dt = 0.01

        def residual() -> float:
            forces = a.spring_forces(body)
            G = a.grasp_matrix(body)
            w_actual = G @ forces
            return float(np.linalg.norm(w_desired - w_actual))

        early_residual = residual()
        for _ in range(200):
            a.update_angles_for_target(body, w_desired)
            f_des = a.allocate_radial_forces(w_desired, body)
            a.compute_ref_radii(body, f_des)
            a.step_actuators(dt)
        late_residual = residual()
        assert (
            late_residual < early_residual or late_residual < 0.5
        ), f"Slow actuators: early {early_residual:.4f}, late {late_residual:.4f}"


# ---------------------------------------------------------------------------
# Repulsive wrench
# ---------------------------------------------------------------------------


class TestRepulsiveWrench:
    """Tests for ActuatorArray.repulsive_wrench (APF local safety)."""

    @pytest.fixture()
    def array3(self) -> ActuatorArray:
        return ActuatorArray(actuator_count=3, standoff=0.05)

    def test_zero_when_all_far(
        self, array3: ActuatorArray, circle: CircleBody
    ) -> None:
        """Wrench is zero when every hazard is beyond d0."""
        w = array3.repulsive_wrench(
            circle,
            lambda pos: (100.0, np.array([100.0, 0.0])),
            np.empty((0, 2)),
            k_rep=5.0,
            d0=0.20,
        )
        np.testing.assert_array_almost_equal(w, [0.0, 0.0, 0.0])

    def test_zero_at_influence_boundary(
        self, array3: ActuatorArray, circle: CircleBody
    ) -> None:
        """Force magnitude is zero exactly at d = d0."""
        d0 = 0.20

        def fn(pos: np.ndarray) -> tuple[float, np.ndarray]:
            return d0, pos + np.array([d0, 0.0])

        w = array3.repulsive_wrench(
            circle, fn, np.empty((0, 2)), k_rep=5.0, d0=d0
        )
        np.testing.assert_array_almost_equal(w, [0.0, 0.0, 0.0])

    def test_nonzero_when_obstacle_close(
        self, array3: ActuatorArray, circle: CircleBody
    ) -> None:
        """Wrench is nonzero when obstacle is within d0."""
        w = array3.repulsive_wrench(
            circle,
            lambda pos: (0.05, pos + np.array([0.05, 0.0])),
            np.empty((0, 2)),
            k_rep=5.0,
            d0=0.20,
        )
        assert np.linalg.norm(w) > 0.0

    def test_formula_symmetric_obstacle_field(
        self, array3: ActuatorArray, circle: CircleBody
    ) -> None:
        """Net force matches k*(d0-d)^2 * N for a uniform field."""
        k_rep = 5.0
        d0 = 0.20
        d_close = 0.08  # well within influence radius

        # Obstacle always at +x direction, d_close away from each actuator
        def fn(pos: np.ndarray) -> tuple[float, np.ndarray]:
            return d_close, pos + np.array([d_close, 0.0])

        w = array3.repulsive_wrench(
            circle, fn, np.empty((0, 2)), k_rep=k_rep, d0=d0
        )
        expected_fx = -3.0 * k_rep * (d0 - d_close) ** 2
        assert w[0] == pytest.approx(expected_fx, rel=1e-6)
        assert w[1] == pytest.approx(0.0, abs=1e-10)
        assert w[2] == pytest.approx(0.0, abs=1e-10)

    def test_direction_away_from_obstacle(
        self, array3: ActuatorArray, circle: CircleBody
    ) -> None:
        """Repulsive force points away from the nearest obstacle."""
        d0 = 0.20
        k_rep = 5.0

        # Obstacle always directly above (+y) each actuator
        def fn_above(pos: np.ndarray) -> tuple[float, np.ndarray]:
            return 0.05, pos + np.array([0.0, 0.05])

        w = array3.repulsive_wrench(
            circle, fn_above, np.empty((0, 2)), k_rep=k_rep, d0=d0
        )
        # Net -y force (away from above obstacle); torque cancels by symmetry
        assert w[1] < 0.0
        assert w[2] == pytest.approx(0.0, abs=1e-10)

    def test_peer_positions_activate_repulsion(
        self, array3: ActuatorArray, circle: CircleBody
    ) -> None:
        """A close peer position triggers repulsion when static obstacles are far."""
        d0 = 0.20
        k_rep = 5.0
        far_fn = lambda pos: (100.0, np.array([100.0, 0.0]))  # noqa: E731

        # With no peers → zero wrench
        w_no_peer = array3.repulsive_wrench(
            circle, far_fn, np.empty((0, 2)), k_rep=k_rep, d0=d0
        )
        np.testing.assert_array_almost_equal(w_no_peer, [0.0, 0.0, 0.0])

        # Peer directly above each actuator at 0.05 m (< d0=0.20) → nonzero
        positions = array3.actuator_positions(circle)
        peers = positions + np.array([[0.0, 0.05]])
        w_with_peer = array3.repulsive_wrench(
            circle, far_fn, peers, k_rep=k_rep, d0=d0
        )
        assert np.linalg.norm(w_with_peer) > 0.0

    def test_empty_other_positions_no_crash(
        self, array3: ActuatorArray, circle: CircleBody
    ) -> None:
        """Passing np.empty((0, 2)) as other_positions must not raise."""
        w = array3.repulsive_wrench(
            circle,
            lambda pos: (100.0, np.array([100.0, 0.0])),
            np.empty((0, 2)),
            k_rep=5.0,
            d0=0.20,
        )
        assert w.shape == (3,)
