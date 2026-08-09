"""DubinsPathFollowingMPC: SE(2) unicycle contouring NMPC (MPCC)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Optional, Sequence

import numpy as np

from arco.config import load_config
from arco.control.mpc.base import MPCTracker
from arco.control.mpc.costs import forward_cone_factor, obstacle_barrier
from arco.control.mpc.reference_path import ReferencePath
from arco.control.mpc.result import MPCStepResult

if TYPE_CHECKING:
    from arco.mapping.occupancy import Occupancy


def _require_casadi() -> Any:
    """Import CasADi or raise an install hint.

    Returns:
        The ``casadi`` module.

    Raises:
        ImportError: If CasADi is not installed.
    """
    try:
        import casadi as ca
    except ImportError as exc:
        raise ImportError(
            "CasADi is required for DubinsPathFollowingMPC. "
            "Install with: pip install arco[mpc]"
        ) from exc
    return ca


def _predicted_xy_from_states(states: np.ndarray) -> list[tuple[float, float]]:
    """Extract ``(x, y)`` samples from an MPC state trajectory matrix.

    Args:
        states: State matrix with shape ``(5, N)`` where rows are
            ``(x, y, θ, v, ω)``.

    Returns:
        Ordered list of ``(x, y)`` tuples along the predicted horizon.
    """
    if states.ndim != 2 or states.shape[0] < 2:
        return []
    return [
        (float(states[0, i]), float(states[1, i]))
        for i in range(int(states.shape[1]))
    ]


@dataclass
class DubinsVehicleLimits:
    """Dynamic limits mirrored from :class:`~arco.guidance.vehicle.DubinsVehicle`.

    Attributes:
        max_speed: Maximum forward speed (m/s).
        min_speed: Minimum forward speed (m/s).
        max_turn_rate: Maximum absolute turn rate (rad/s).
        max_acceleration: Maximum linear acceleration (m/s²).
        max_turn_rate_dot: Maximum turn-rate derivative (rad/s²).
    """

    max_speed: float
    min_speed: float
    max_turn_rate: float
    max_acceleration: float
    max_turn_rate_dot: float


@dataclass
class PathFollowingMPCConfig:
    """Tunable weights and horizon for Dubins path-following MPCC.

    The controller is a Model Predictive Contouring Controller: the path
    parameter ``s`` advances with its own *virtual progress speed*
    decision variable, and the position error is split into a lateral
    **contouring** error and a longitudinal **lag** error, both evaluated
    at the reference point ``p(s_k)``.

    Attributes:
        horizon_step_count: Prediction horizon length (steps).
        dt: Discretization step for the prediction model (s).  Must match
            the closed-loop control period for the first predicted state
            to be a valid command target.
        cruise_speed: Nominal progress speed on straights (m/s).
        weight_contour: Lateral (contouring) error weight applied outside
            :attr:`contour_deadzone`.
        weight_heading: Heading alignment weight (smooth 2π-periodic cost).
        weight_progress: Linear progress reward per meter of arc-length
            advancement.  The virtual speed is hard-capped by the
            curve-limited reference speed, so this only decides how
            strongly advancement pays against tracking costs.
        weight_lag: Lag-error weight.  Structural in MPCC: it couples the
            virtual progress ``s`` to the vehicle position.  Larger values
            keep ``p(s)`` glued to the vehicle; ``0`` decouples ``s`` and
            is rejected at construction time.
        weight_control: Control-effort weight on ``(a, ω̇)``.
        weight_obstacle: Soft obstacle-barrier weight.
        obstacle_barrier_power: Barrier exponent.
        weight_terminal: Terminal contouring / heading weight.
        contour_deadzone: Lateral free band (m).  Contouring cost is zero
            for ``|e_lat| ≤ contour_deadzone`` and quadratic on the excess.
        max_solver_iter_count: IPOPT iteration budget.
    """

    horizon_step_count: int = 20
    dt: float = 0.05
    cruise_speed: float = 0.36
    weight_contour: float = 10.0
    weight_heading: float = 2.0
    weight_progress: float = 1.0
    weight_lag: float = 4.0
    weight_control: float = 0.1
    weight_obstacle: float = 50.0
    obstacle_barrier_power: float = 4.0
    weight_terminal: float = 20.0
    contour_deadzone: float = 0.0
    max_solver_iter_count: int = 80

    @staticmethod
    def create_from_config(
        cruise_speed: Optional[float] = None,
    ) -> PathFollowingMPCConfig:
        """Load defaults from ``config/mpc.yml``.

        Args:
            cruise_speed: Optional override for the cruise speed (m/s).

        Returns:
            Configured :class:`PathFollowingMPCConfig` instance.
        """
        cfg = load_config("mpc")
        horizon = cfg.get("horizon", {})
        weights = cfg.get("weights", {})
        barrier = cfg.get("obstacle_barrier", {})
        solver = cfg.get("solver", {})
        cruise = 0.36
        if cruise_speed is not None:
            cruise = float(cruise_speed)
        elif "cruise_speed" in cfg:
            cruise = float(cfg["cruise_speed"])
        return PathFollowingMPCConfig(
            horizon_step_count=int(horizon.get("step_count", 20)),
            dt=float(horizon.get("dt", 0.05)),
            cruise_speed=cruise,
            weight_contour=float(weights.get("contour", 10.0)),
            weight_heading=float(weights.get("heading", 2.0)),
            weight_progress=float(weights.get("progress", 1.0)),
            weight_lag=float(weights.get("lag", 4.0)),
            weight_control=float(weights.get("control", 0.1)),
            weight_obstacle=float(weights.get("obstacle", 50.0)),
            weight_terminal=float(weights.get("terminal", 20.0)),
            contour_deadzone=float(weights.get("contour_deadzone", 0.0)),
            obstacle_barrier_power=float(barrier.get("power", 4.0)),
            max_solver_iter_count=int(solver.get("max_iter_count", 80)),
        )

    def with_horizon_overrides(
        self,
        *,
        step_count: int | None = None,
        dt: float | None = None,
    ) -> PathFollowingMPCConfig:
        """Return a copy with optional horizon overrides applied.

        Args:
            step_count: Optional new prediction horizon length (steps).
            dt: Optional new discretization step (s).

        Returns:
            A new :class:`PathFollowingMPCConfig` with the requested
            horizon fields replaced; other fields are unchanged.
        """
        return replace(
            self,
            horizon_step_count=(
                int(step_count)
                if step_count is not None
                else self.horizon_step_count
            ),
            dt=float(dt) if dt is not None else self.dt,
        )

    def with_weight_overrides(
        self,
        *,
        contour: float | None = None,
        heading: float | None = None,
        progress: float | None = None,
        lag: float | None = None,
        control: float | None = None,
        obstacle: float | None = None,
        terminal: float | None = None,
        contour_deadzone: float | None = None,
    ) -> PathFollowingMPCConfig:
        """Return a copy with optional cost-weight overrides applied.

        Args:
            contour: Optional lateral / contouring weight.
            heading: Optional heading-alignment weight.
            progress: Optional virtual-progress speed-tracking weight.
            lag: Optional lag-error weight (must stay positive).
            control: Optional control-effort weight.
            obstacle: Optional obstacle-barrier weight.
            terminal: Optional terminal cost weight.
            contour_deadzone: Optional free lateral band (m).

        Returns:
            A new :class:`PathFollowingMPCConfig` with the requested
            weight fields replaced; other fields are unchanged.
        """
        return replace(
            self,
            weight_contour=(
                float(contour) if contour is not None else self.weight_contour
            ),
            weight_heading=(
                float(heading) if heading is not None else self.weight_heading
            ),
            weight_progress=(
                float(progress)
                if progress is not None
                else self.weight_progress
            ),
            weight_lag=(float(lag) if lag is not None else self.weight_lag),
            weight_control=(
                float(control) if control is not None else self.weight_control
            ),
            weight_obstacle=(
                float(obstacle)
                if obstacle is not None
                else self.weight_obstacle
            ),
            weight_terminal=(
                float(terminal)
                if terminal is not None
                else self.weight_terminal
            ),
            contour_deadzone=(
                float(contour_deadzone)
                if contour_deadzone is not None
                else self.contour_deadzone
            ),
        )


class DubinsPathFollowingMPC(MPCTracker):
    """Receding-horizon contouring NMPC (MPCC) for Dubins vehicles.

    Follows the Lam / Liniger MPCC structure: the path parameter ``s`` is
    advanced by a bounded **virtual progress speed** decision variable,
    and the position error at ``p(s_k)`` is split into a lateral
    contouring error and a longitudinal lag error.  The lag term couples
    the virtual progress to the vehicle, so no projection heuristics or
    monotonicity constraints are needed inside the NLP.

    Mathematical formulation, symbols, and differences from classical
    racing MPCC are documented in ``docs/control_mpcc.md``.

    Raises:
        ValueError: If ``config.weight_lag`` is not strictly positive —
            with a zero lag weight the virtual progress decouples from
            the vehicle and the contouring errors lose meaning.
    """

    _OBSTACLE_SAMPLE_COUNT = 5
    # Reference resampling resolution (m) and sample-count bounds.  The
    # previous fixed 200-sample grid smeared corners on ~900 m city paths.
    _PATH_INTERP_RESOLUTION_M = 2.0
    _PATH_INTERP_MIN_COUNT = 200
    _PATH_INTERP_MAX_COUNT = 1500

    def __init__(
        self,
        *,
        vehicle_limits: DubinsVehicleLimits,
        config: PathFollowingMPCConfig,
        occupancy: Occupancy | None = None,
    ) -> None:
        """Initialize the Dubins path-following MPCC.

        Args:
            vehicle_limits: Speed / turn-rate / acceleration limits.
            config: Horizon and cost weights.
            occupancy: Optional occupancy map for obstacle barriers.

        Raises:
            ImportError: If the optional CasADi dependency is missing.
            ValueError: If ``config.weight_lag <= 0``.
        """
        # Eager dependency check so missing CasADi fails at construction.
        self._ca = _require_casadi()
        if config.weight_lag <= 0.0:
            raise ValueError(
                "PathFollowingMPCConfig.weight_lag must be > 0: the lag "
                "error is what couples the virtual progress s to the "
                "vehicle in the MPCC formulation."
            )
        self.vehicle_limits = vehicle_limits
        self.config = config
        self._occupancy = occupancy
        self._reference: ReferencePath | None = None
        self._progress = 0.0
        self._warm_x: np.ndarray | None = None
        self._warm_u: np.ndarray | None = None
        self._warm_s: np.ndarray | None = None
        self._warm_vs: np.ndarray | None = None
        self._last_speed_cmd = config.cruise_speed
        self._last_turn_rate_cmd = 0.0
        self._nlp: dict[str, Any] | None = None
        self._path_s: np.ndarray | None = None
        self._path_x: np.ndarray | None = None
        self._path_y: np.ndarray | None = None
        self._path_heading: np.ndarray | None = None
        self._path_kappa: np.ndarray | None = None
        self._interp_id = 0

    def set_reference(self, waypoints: Sequence[tuple[float, float]]) -> None:
        """Set or replace the reference path.

        The polyline is extended by one prediction-horizon length of
        straight "runway" along the final tangent so the NLP feasible set
        has no hard corner at the goal (progress bounds pinching ``S``
        against its cap caused end-of-path solver failures).

        Args:
            waypoints: Ordered ``(x, y)`` waypoints in world frame.
        """
        pts = [(float(x), float(y)) for x, y in waypoints]
        if len(pts) >= 2:
            pad = max(
                2.0,
                self.config.cruise_speed
                * self.config.dt
                * self.config.horizon_step_count,
            )
            dx = pts[-1][0] - pts[-2][0]
            dy = pts[-1][1] - pts[-2][1]
            norm = math.hypot(dx, dy)
            if norm > 1e-9:
                pts.append(
                    (
                        pts[-1][0] + pad * dx / norm,
                        pts[-1][1] + pad * dy / norm,
                    )
                )
        self._reference = ReferencePath(pts)
        sample_count = int(
            np.clip(
                math.ceil(
                    self._reference.total_length
                    / self._PATH_INTERP_RESOLUTION_M
                ),
                self._PATH_INTERP_MIN_COUNT,
                self._PATH_INTERP_MAX_COUNT,
            )
        )
        (
            self._path_s,
            self._path_x,
            self._path_y,
            self._path_heading,
            self._path_kappa,
        ) = self._reference.sample(sample_count)
        self._progress = 0.0
        self._warm_x = None
        self._warm_u = None
        self._warm_s = None
        self._warm_vs = None
        self._nlp = None

    def step(
        self,
        pose: tuple[float, float, float],
        *,
        speed: float,
        turn_rate: float,
        dt: float,
    ) -> MPCStepResult:
        """Solve one NMPC step and return commands plus diagnostics.

        Args:
            pose: Current vehicle pose ``(x, y, heading)``.
            speed: Current forward speed (m/s).
            turn_rate: Current turn rate (rad/s).
            dt: Control period until the next call (s).  The internal
                prediction model uses :attr:`PathFollowingMPCConfig.dt`;
                configure them equal for consistent closed-loop tracking.

        Returns:
            Structured command and solver diagnostics.

        Raises:
            RuntimeError: If :meth:`set_reference` has not been called.
        """
        del dt  # Commands are rate-limited targets; horizon uses config.dt.
        if self._reference is None:
            raise RuntimeError(
                "DubinsPathFollowingMPC.set_reference() must be called "
                "before step()."
            )

        if not self._state_is_finite(pose, speed, turn_rate):
            return self._safe_stop_result(
                pose,
                speed,
                turn_rate,
                status="invalid_state",
                solve_time_s=0.0,
            )

        # Local projection around the current contouring progress.  A global
        # nearest-point search can jump by a full city block when two road
        # corridors of the route pass near each other.  Progress never
        # rewinds: recovery arcs must catch up to s, not reset it (the lag
        # cost pulls the vehicle forward toward p(s)).
        horizon_m = (
            self.config.cruise_speed
            * self.config.dt
            * self.config.horizon_step_count
        )
        project_window = max(float(horizon_m), 30.0)
        s_proj, e_lat, e_head = self._reference.project(
            pose,
            s_hint=self._progress,
            window=project_window,
        )
        self._progress = float(
            np.clip(
                max(s_proj, self._progress),
                0.0,
                self._reference.total_length,
            )
        )

        obstacles = self._collect_obstacle_samples(pose)
        clearance = float(getattr(self._occupancy, "clearance", 1.0) or 1.0)
        # Fixed obstacle-parameter slots when occupancy is configured.
        obstacle_slot_count = (
            self._OBSTACLE_SAMPLE_COUNT if self._occupancy is not None else 0
        )
        while len(obstacles) < obstacle_slot_count:
            # Far dummy points so unused slots do not affect the barrier.
            obstacles.append((1e6, 1e6))
        obstacles = obstacles[:obstacle_slot_count]
        preview_cruise = self._preview_cruise_speed(clearance)

        t0 = time.perf_counter()
        try:
            sol = self._solve(
                pose=pose,
                speed=speed,
                turn_rate=turn_rate,
                progress=self._progress,
                obstacles=obstacles,
                clearance=clearance,
                obstacle_slot_count=obstacle_slot_count,
                preview_cruise=preview_cruise,
            )
            solve_time_s = time.perf_counter() - t0
        except Exception as exc:  # noqa: BLE001 — solver robustness
            solve_time_s = time.perf_counter() - t0
            return self._safe_stop_result(
                pose,
                speed,
                turn_rate,
                status=f"solver_exception:{type(exc).__name__}",
                solve_time_s=solve_time_s,
                cross_track_error=e_lat,
                heading_error=e_head,
            )

        if sol is None:
            return self._safe_stop_result(
                pose,
                speed,
                turn_rate,
                status="solve_failed",
                solve_time_s=solve_time_s,
                cross_track_error=e_lat,
                heading_error=e_head,
            )

        speed_cmd = float(sol["speed_cmd"])
        turn_rate_cmd = float(sol["turn_rate_cmd"])
        self._last_speed_cmd = speed_cmd
        self._last_turn_rate_cmd = turn_rate_cmd
        self._progress = float(sol["progress"])
        self._warm_x = sol["X"]
        self._warm_u = sol["U"]
        self._warm_s = sol["S"]
        self._warm_vs = sol["VS"]

        predicted_xy = _predicted_xy_from_states(sol["X"])
        return MPCStepResult(
            speed_cmd=speed_cmd,
            turn_rate_cmd=turn_rate_cmd,
            cross_track_error=float(e_lat),
            heading_error=float(e_head),
            progress=float(sol["progress"]),
            predicted_clearance_min=float(sol["clearance_min"]),
            solver_success=True,
            solver_status=str(sol["status"]),
            solve_time_s=float(solve_time_s),
            cost=float(sol["cost"]),
            predicted_xy=predicted_xy,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _state_is_finite(
        pose: tuple[float, float, float], speed: float, turn_rate: float
    ) -> bool:
        values = [pose[0], pose[1], pose[2], speed, turn_rate]
        return all(math.isfinite(float(v)) for v in values)

    def _safe_stop_result(
        self,
        pose: tuple[float, float, float],
        speed: float,
        turn_rate: float,
        *,
        status: str,
        solve_time_s: float,
        cross_track_error: float = 0.0,
        heading_error: float = 0.0,
    ) -> MPCStepResult:
        """Return a decelerating fallback command without PP/APF."""
        limits = self.vehicle_limits
        dt = self.config.dt
        if status == "solve_failed" and self._warm_u is not None:
            speed_cmd = float(self._last_speed_cmd)
            turn_rate_cmd = float(self._last_turn_rate_cmd)
        else:
            speed_cmd = max(
                limits.min_speed, float(speed) - limits.max_acceleration * dt
            )
            turn_rate_cmd = float(turn_rate)  # ω̇ = 0

        if self._reference is not None and self._state_is_finite(
            pose, speed, turn_rate
        ):
            horizon_m = (
                self.config.cruise_speed
                * self.config.dt
                * self.config.horizon_step_count
            )
            _, cross_track_error, heading_error = self._reference.project(
                pose,
                s_hint=self._progress,
                window=max(float(horizon_m), 30.0),
            )

        predicted_xy: list[tuple[float, float]] = []
        if self._warm_x is not None and self._warm_x.ndim == 2:
            predicted_xy = _predicted_xy_from_states(self._warm_x)
        elif self._state_is_finite(pose, speed, turn_rate):
            predicted_xy = [(float(pose[0]), float(pose[1]))]
        return MPCStepResult(
            speed_cmd=float(speed_cmd),
            turn_rate_cmd=float(turn_rate_cmd),
            cross_track_error=float(cross_track_error),
            heading_error=float(heading_error),
            progress=float(self._progress),
            predicted_clearance_min=float("inf"),
            solver_success=False,
            solver_status=status,
            solve_time_s=float(solve_time_s),
            cost=float("inf"),
            predicted_xy=predicted_xy,
        )

    def _look_ahead_distance(self) -> float:
        horizon_distance = (
            self.config.cruise_speed
            * self.config.dt
            * self.config.horizon_step_count
        )
        # Preview beyond the short prediction horizon at low cruise speeds.
        return max(horizon_distance * 1.5, 2.0)

    def _preview_cruise_speed(self, clearance: float) -> float:
        """Reduce cruise when path-ahead clearance is tight.

        Evaluates ``nearest_obstacle`` along the reference ahead of the
        current progress so the optimizer slows before pinch points enter
        the short prediction horizon.
        """
        cruise = float(self.config.cruise_speed)
        if (
            self._occupancy is None
            or self._reference is None
            or not hasattr(self._occupancy, "nearest_obstacle")
            or clearance <= 0.0
        ):
            return cruise
        look_ahead = self._look_ahead_distance()
        min_dist = float("inf")
        for alpha in np.linspace(0.0, 1.0, 12):
            s = min(
                self._progress + alpha * look_ahead,
                self._reference.total_length,
            )
            x_ref, y_ref = self._reference.position(s)
            dist, _ = self._occupancy.nearest_obstacle(
                np.array([x_ref, y_ref], dtype=float)
            )
            min_dist = min(min_dist, float(dist))
        if min_dist >= clearance:
            return cruise
        # Linearly taper cruise as path-ahead clearance collapses.
        scale = max(0.15, min_dist / clearance)
        return max(self.vehicle_limits.min_speed, cruise * scale)

    def _collect_obstacle_samples(
        self, pose: tuple[float, float, float]
    ) -> list[tuple[float, float]]:
        if self._occupancy is None or not hasattr(
            self._occupancy, "nearest_obstacle"
        ):
            return []
        samples: list[tuple[float, float]] = []
        pts = [np.array([pose[0], pose[1]], dtype=float)]
        if self._reference is not None:
            look_ahead = self._look_ahead_distance()
            for alpha in np.linspace(0.0, 1.0, self._OBSTACLE_SAMPLE_COUNT):
                s = min(
                    self._progress + alpha * look_ahead,
                    self._reference.total_length,
                )
                x_ref, y_ref = self._reference.position(s)
                pts.append(np.array([x_ref, y_ref], dtype=float))
        seen: set[tuple[float, float]] = set()
        for pt in pts:
            _dist, nearest = self._occupancy.nearest_obstacle(pt)
            key = (round(float(nearest[0]), 4), round(float(nearest[1]), 4))
            if key in seen:
                continue
            seen.add(key)
            samples.append((float(nearest[0]), float(nearest[1])))
        return samples

    def _reference_rollout_init(
        self,
        x0_val: np.ndarray,
        progress: float,
        preview_cruise: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build an NLP initial guess that rolls forward along the reference.

        The rollout accelerates from the current speed toward the
        curve-limited reference speed and places the predicted poses on
        the path itself.  Seeding IPOPT inside this "moving" basin is what
        lets the solver escape the parked equilibrium at sharp kinks,
        where a constant-heading or all-stopped guess converges to a
        full-stop local minimum.

        Args:
            x0_val: Current state ``(x, y, θ, v, ω)``.
            progress: Current path progress ``s₀`` (m).
            preview_cruise: Clearance-limited cruise speed (m/s).

        Returns:
            ``(X_init, U_init, S_init, VS_init)`` initial-guess arrays.
        """
        assert self._reference is not None
        cfg = self.config
        limits = self.vehicle_limits
        n = cfg.horizon_step_count
        total = self._reference.total_length

        X_init = np.zeros((5, n + 1))
        U_init = np.zeros((2, n))
        S_init = np.zeros((1, n + 1))
        VS_init = np.zeros((1, n))
        X_init[:, 0] = x0_val
        S_init[0, 0] = progress

        v = float(np.clip(x0_val[3], limits.min_speed, limits.max_speed))
        s = float(progress)
        heading_prev = float(x0_val[2])
        for k in range(n):
            kappa = self._reference.curvature(s)
            v_curve = limits.max_turn_rate / max(abs(kappa), 1e-3)
            v_target = float(
                np.clip(
                    min(preview_cruise, v_curve),
                    limits.min_speed,
                    limits.max_speed,
                )
            )
            dv = float(
                np.clip(
                    v_target - v,
                    -limits.max_acceleration * cfg.dt,
                    limits.max_acceleration * cfg.dt,
                )
            )
            v = v + dv
            s = min(s + v * cfg.dt, total)
            x_ref, y_ref = self._reference.position(s)
            heading_ref = self._reference.heading(s)
            # Continuous heading guess (unwrap against the previous one).
            heading_ref = heading_prev + math.atan2(
                math.sin(heading_ref - heading_prev),
                math.cos(heading_ref - heading_prev),
            )
            omega = float(
                np.clip(
                    (heading_ref - heading_prev) / cfg.dt,
                    -limits.max_turn_rate,
                    limits.max_turn_rate,
                )
            )
            heading_prev = heading_ref
            X_init[0, k + 1] = x_ref
            X_init[1, k + 1] = y_ref
            X_init[2, k + 1] = heading_ref
            X_init[3, k + 1] = v
            X_init[4, k + 1] = omega
            VS_init[0, k] = v
            S_init[0, k + 1] = s
        return X_init, U_init, S_init, VS_init

    def _build_nlp(self, obstacle_count: int) -> dict[str, Any]:
        ca = self._ca
        cfg = self.config
        limits = self.vehicle_limits
        n = cfg.horizon_step_count
        dt = cfg.dt
        assert self._reference is not None
        assert self._path_s is not None

        s_grid = np.asarray(self._path_s, dtype=float)
        heading_unwrapped = np.unwrap(np.asarray(self._path_heading))
        self._interp_id += 1
        tag = f"mpc{self._interp_id}"
        # Cubic B-spline interpolants: piecewise-linear reference lookups
        # have discontinuous gradients exactly at path kinks, which stalls
        # IPOPT (Maximum_Iterations_Exceeded) right where tracking is
        # hardest.  Smooth splines also round the polyline at the ~2 m
        # sample scale, which is desirable for a vehicle-feasible target.
        fx = ca.interpolant(f"{tag}_x", "bspline", [s_grid], self._path_x)
        fy = ca.interpolant(f"{tag}_y", "bspline", [s_grid], self._path_y)
        fth = ca.interpolant(
            f"{tag}_th", "bspline", [s_grid], heading_unwrapped
        )
        fk = ca.interpolant(f"{tag}_k", "bspline", [s_grid], self._path_kappa)

        opti = ca.Opti()
        # State: [px, py, theta, v, omega]
        X = opti.variable(5, n + 1)
        U = opti.variable(2, n)  # [a, omega_dot]
        S = opti.variable(1, n + 1)  # path parameter (arc length)
        VS = opti.variable(1, n)  # virtual progress speed ds/dt

        x0 = opti.parameter(5)
        s0 = opti.parameter(1)
        clearance_p = opti.parameter(1)
        cruise_p = opti.parameter(1)
        obs = opti.parameter(2, obstacle_count) if obstacle_count > 0 else None

        opti.subject_to(X[:, 0] == x0)
        opti.subject_to(S[0, 0] == s0)

        def _stage_errors(k: int) -> tuple[Any, Any, Any]:
            """Contouring / lag / heading errors at stage *k*."""
            x_ref = fx(S[0, k])
            y_ref = fy(S[0, k])
            th_ref = fth(S[0, k])
            dx = X[0, k] - x_ref
            dy = X[1, k] - y_ref
            e_contour = -dx * ca.sin(th_ref) + dy * ca.cos(th_ref)
            e_lag = dx * ca.cos(th_ref) + dy * ca.sin(th_ref)
            e_head = X[2, k] - th_ref
            return e_contour, e_lag, e_head

        cost = 0
        for k in range(n):
            v = X[3, k]
            theta = X[2, k]
            omega = X[4, k]
            a = U[0, k]
            omega_dot = U[1, k]
            v_s = VS[0, k]

            e_contour, e_lag, e_head = _stage_errors(k)
            e_head_cost = ca.sin(e_head) ** 2 + (1.0 - ca.cos(e_head)) ** 2

            kappa = fk(S[0, k])
            # Curve-limited progress cap, written as two smooth
            # inequalities (no fmin/fabs kinks in the constraint set):
            #   v_s ≤ cruise   and   v_s·|κ| ≤ ω_max  ⇔  v_s ≤ ω_max/|κ|.
            opti.subject_to(v_s <= cruise_p)
            opti.subject_to(
                v_s * ca.sqrt(kappa**2 + 1e-6) <= limits.max_turn_rate
            )

            # Free lateral band: only the excess beyond the deadzone is
            # penalized.  The zero-deadzone default uses the plain smooth
            # quadratic (no fabs/fmax kink at e_contour = 0).
            if cfg.contour_deadzone > 0.0:
                e_c_excess = ca.fmax(
                    ca.fabs(e_contour) - cfg.contour_deadzone, 0
                )
                cost += cfg.weight_contour * e_c_excess**2
            else:
                cost += cfg.weight_contour * e_contour**2
            cost += cfg.weight_lag * e_lag**2
            cost += cfg.weight_heading * e_head_cost
            # Progress: LINEAR advancement reward (classical MPCC).  A
            # quadratic (v_s − v_ref)² speed-matching term is a trap: at a
            # sharp kink v_ref(s) is small, so parking at the kink costs
            # almost nothing while accelerating away looks expensive over
            # the horizon — the solver then prefers a permanent full stop.
            # A linear reward makes advancement pay everywhere; the hard
            # v_s ≤ v_ref cap above keeps corner braking feed-forward.
            cost -= cfg.weight_progress * v_s * dt
            cost += cfg.weight_control * (a**2 + omega_dot**2)

            if obs is not None:
                px = X[0, k]
                py = X[1, k]
                for j in range(obstacle_count):
                    ox = obs[0, j]
                    oy = obs[1, j]
                    dist = ca.sqrt((px - ox) ** 2 + (py - oy) ** 2 + 1e-9)
                    cone = forward_cone_factor(px, py, theta, ox, oy)
                    cost += obstacle_barrier(
                        dist,
                        clearance_p,
                        cfg.weight_obstacle,
                        cfg.obstacle_barrier_power,
                        cone,
                    )

            v_next = v + a * dt
            omega_next = omega + omega_dot * dt
            opti.subject_to(X[0, k + 1] == X[0, k] + v * ca.cos(theta) * dt)
            opti.subject_to(X[1, k + 1] == X[1, k] + v * ca.sin(theta) * dt)
            opti.subject_to(X[2, k + 1] == theta + omega * dt)
            opti.subject_to(X[3, k + 1] == v_next)
            opti.subject_to(X[4, k + 1] == omega_next)
            # Virtual progress dynamics: s advances with its own bounded
            # speed decision variable (classical MPCC progress law).
            opti.subject_to(S[0, k + 1] == S[0, k] + v_s * dt)
            opti.subject_to(opti.bounded(0.0, v_s, limits.max_speed))

            opti.subject_to(
                opti.bounded(limits.min_speed, v_next, limits.max_speed)
            )
            opti.subject_to(
                opti.bounded(
                    -limits.max_turn_rate, omega_next, limits.max_turn_rate
                )
            )
            opti.subject_to(
                opti.bounded(
                    -limits.max_acceleration, a, limits.max_acceleration
                )
            )
            opti.subject_to(
                opti.bounded(
                    -limits.max_turn_rate_dot,
                    omega_dot,
                    limits.max_turn_rate_dot,
                )
            )
            opti.subject_to(
                opti.bounded(0.0, S[0, k], self._reference.total_length)
            )

        e_contour_n, e_lag_n, e_head_n = _stage_errors(n)
        e_head_n_cost = ca.sin(e_head_n) ** 2 + (1.0 - ca.cos(e_head_n)) ** 2
        if cfg.contour_deadzone > 0.0:
            e_c_n_excess = ca.fmax(
                ca.fabs(e_contour_n) - cfg.contour_deadzone, 0.0
            )
            e_c_n_cost = e_c_n_excess**2
        else:
            e_c_n_cost = e_contour_n**2
        cost += cfg.weight_terminal * (e_c_n_cost + e_head_n_cost)
        cost += cfg.weight_lag * e_lag_n**2
        opti.subject_to(
            opti.bounded(limits.min_speed, X[3, n], limits.max_speed)
        )
        opti.subject_to(
            opti.bounded(-limits.max_turn_rate, X[4, n], limits.max_turn_rate)
        )
        opti.subject_to(
            opti.bounded(0.0, S[0, n], self._reference.total_length)
        )

        opti.minimize(cost)
        opti.solver(
            "ipopt",
            {
                "ipopt.print_level": 0,
                "print_time": 0,
                "ipopt.max_iter": int(cfg.max_solver_iter_count),
                "ipopt.sb": "yes",
                "ipopt.tol": 1e-4,
                "ipopt.acceptable_tol": 1e-2,
                "ipopt.acceptable_iter": 5,
                "ipopt.warm_start_init_point": "yes",
            },
        )
        return {
            "opti": opti,
            "X": X,
            "U": U,
            "S": S,
            "VS": VS,
            "x0": x0,
            "s0": s0,
            "obs": obs,
            "clearance": clearance_p,
            "cruise": cruise_p,
            "obstacle_count": obstacle_count,
            "cost": cost,
        }

    def _ensure_nlp(self, obstacle_count: int) -> dict[str, Any]:
        if (
            self._nlp is None
            or int(self._nlp["obstacle_count"]) != obstacle_count
        ):
            # Unique interpolant names per rebuild to avoid CasADi collisions.
            self._nlp = self._build_nlp(obstacle_count)
        return self._nlp

    def _solve(
        self,
        *,
        pose: tuple[float, float, float],
        speed: float,
        turn_rate: float,
        progress: float,
        obstacles: list[tuple[float, float]],
        clearance: float,
        obstacle_slot_count: int,
        preview_cruise: float,
    ) -> dict[str, Any] | None:
        cfg = self.config
        limits = self.vehicle_limits
        n = cfg.horizon_step_count
        nlp = self._ensure_nlp(obstacle_slot_count)

        opti = nlp["opti"]
        X = nlp["X"]
        U = nlp["U"]
        S = nlp["S"]
        VS = nlp["VS"]

        x0_val = np.array(
            [pose[0], pose[1], pose[2], speed, turn_rate], dtype=float
        )
        opti.set_value(nlp["x0"], x0_val)
        opti.set_value(nlp["s0"], progress)
        opti.set_value(nlp["clearance"], max(clearance, 1e-3))
        opti.set_value(nlp["cruise"], float(preview_cruise))

        if obstacle_slot_count > 0 and nlp["obs"] is not None:
            obs_mat = np.array(obstacles, dtype=float).T
            opti.set_value(nlp["obs"], obs_mat)

        # Shifted warm start while the previous solution keeps moving.  A
        # stalled warm start (near-zero progress over the horizon while
        # the reference ahead allows motion) would re-seed IPOPT inside
        # the parked local minimum forever — use a fresh reference rollout
        # instead so the solver can compare against the moving basin.
        warm_available = (
            self._warm_x is not None
            and self._warm_x.shape == (5, n + 1)
            and self._warm_vs is not None
        )
        warm_ok = False
        if warm_available:
            warm_advance = float(self._warm_s[0, -1] - self._warm_s[0, 0])
            horizon_advance = float(preview_cruise) * cfg.dt * n
            near_goal = (
                self._reference.total_length - progress
            ) < horizon_advance
            warm_ok = (
                warm_advance > max(1.0, 0.1 * horizon_advance)
                or preview_cruise <= limits.min_speed + 1e-9
                or near_goal
            )
        if warm_ok:
            X_init = np.hstack([self._warm_x[:, 1:], self._warm_x[:, -1:]])
            U_init = np.hstack([self._warm_u[:, 1:], self._warm_u[:, -1:]])
            S_init = np.hstack([self._warm_s[:, 1:], self._warm_s[:, -1:]])
            VS_init = np.hstack([self._warm_vs[:, 1:], self._warm_vs[:, -1:]])
            X_init[:, 0] = x0_val
            S_init[0, 0] = progress
            # Keep the shifted progress column consistent with s0.
            S_init[0, 1:] = np.maximum(S_init[0, 1:], progress)
        else:
            X_init, U_init, S_init, VS_init = self._reference_rollout_init(
                x0_val, progress, preview_cruise
            )

        S_init = np.clip(S_init, 0.0, self._reference.total_length)
        VS_init = np.clip(VS_init, 0.0, limits.max_speed)
        opti.set_initial(X, X_init)
        opti.set_initial(U, U_init)
        opti.set_initial(S, S_init)
        opti.set_initial(VS, VS_init)

        try:
            sol = opti.solve()
        except RuntimeError:
            return None

        X_opt = np.array(sol.value(X), dtype=float)
        U_opt = np.array(sol.value(U), dtype=float)
        S_opt = np.array(sol.value(S), dtype=float).reshape(1, n + 1)
        VS_opt = np.array(sol.value(VS), dtype=float).reshape(1, n)
        speed_cmd = float(
            np.clip(X_opt[3, 1], limits.min_speed, limits.max_speed)
        )
        turn_rate_cmd = float(
            np.clip(X_opt[4, 1], -limits.max_turn_rate, limits.max_turn_rate)
        )

        clearance_min = float("inf")
        if self._occupancy is not None and hasattr(
            self._occupancy, "nearest_obstacle"
        ):
            dists = []
            for k in range(n + 1):
                pt = np.array([X_opt[0, k], X_opt[1, k]], dtype=float)
                dist, _ = self._occupancy.nearest_obstacle(pt)
                dists.append(float(dist))
            clearance_min = min(dists) if dists else float("inf")

        status = "Solve_Succeeded"
        try:
            status = str(sol.stats().get("return_status", status))
        except Exception:  # noqa: BLE001
            pass

        return {
            "speed_cmd": speed_cmd,
            "turn_rate_cmd": turn_rate_cmd,
            "progress": float(S_opt[0, 1]),
            "clearance_min": clearance_min,
            "status": status,
            "cost": float(sol.value(nlp["cost"])),
            "X": X_opt,
            "U": U_opt,
            "S": S_opt,
            "VS": VS_opt,
        }
