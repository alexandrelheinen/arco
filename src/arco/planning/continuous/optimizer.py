"""TrajectoryOptimizer: two-stage trajectory refinement for global paths."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from arco.mapping.occupancy import Occupancy


@dataclass
class TrajectoryResult:
    """Result of a trajectory optimization run.

    Attributes:
        states: Ordered list of N+1 position arrays ``(x, y)`` (or higher
            dimensional) from start to goal, including the fixed endpoints.
        commands: List of N control-command arrays, one per segment.  Each
            entry is the output of the inverse-kinematics callable (or an
            estimated command when no IK is provided).
        durations: List of N positive segment traversal times (seconds).
        cost: Scalar value of the composite cost function at the optimized
            solution.
        is_feasible: ``True`` when the optimized trajectory satisfies all
            dynamic constraints (speed bounds, feasibility callable).
            ``False`` indicates at least one waypoint or segment violates
            the model limits; the vehicle should stall instead of executing
            this trajectory.
    """

    states: List[np.ndarray] = field(default_factory=list)
    commands: List[np.ndarray] = field(default_factory=list)
    durations: List[float] = field(default_factory=list)
    cost: float = 0.0
    is_feasible: bool = True


class TrajectoryOptimizer:
    """Model-agnostic two-stage trajectory optimizer.

    Refines a reference path from a global planner (RRT*, SST, …) into a
    time-optimal trajectory that minimizes a five-term composite cost while
    staying close to the reference path and avoiding obstacles.

    **Stage 1 — Initialisation (inverse kinematics).**
    An initial guess is built by placing each interior waypoint directly on
    the reference path and setting the segment duration proportional to the
    straight-line distance at cruise speed (with a relaxation factor α).
    If an ``inverse_kinematics`` callable is supplied, it is used to compute
    the initial control commands; otherwise a straight-line approximation is
    used.

    **Stage 2 — Local refinement (scipy.optimize.minimize).**
    The Stage-1 candidate is passed to L-BFGS-B (or another *method*)
    which tightens the solution by jointly optimising segment durations and
    interior waypoint positions.

    The composite cost is:

    .. code-block:: text

        J = w_time · T²
          + w_deviation · Σ |pᵢ − refᵢ|²
          + w_velocity · Σ (|pᵢ − pᵢ₋₁| / tᵢ − v_cruise)²
          + w_collision · Σ max(0, clearance − dist(pᵢ, obstacles))²
          + w_dynamics · Σ [max(0, vᵢ − max_speed)² + max(0, min_speed − vᵢ)²]

    where *T = Σ tᵢ* is the total traversal time and the dynamics term
    penalises implied segment speeds that exceed :attr:`max_speed` or fall
    below :attr:`min_speed`.  With a large *weight_dynamics* this penalty
    acts as a near-hard constraint during optimisation.

    After optimisation :meth:`optimize` checks every segment's implied speed
    and every waypoint via the optional *feasibility* callable.  If any
    violation is detected :attr:`~TrajectoryResult.is_feasible` is set to
    ``False`` on the returned result.  Callers must inspect this flag and
    stall the vehicle rather than execute an infeasible trajectory.

    The trajectory is discretized by a progress variable *s* that advances
    linearly from 0 to *N* (one unit per segment).  The bijection between
    progress and time is ``t(s) = Σ_{j<⌊s⌋} tⱼ + frac(s) · t_{⌊s⌋}``.

    Args:
        occupancy: Occupancy map used for the collision penalty term (must
            implement :meth:`~arco.mapping.Occupancy.nearest_obstacle`).
        cruise_speed: Target traversal speed (world units / s).  Drives
            the velocity penalty and the initial time estimate.  Must be
            positive.
        weight_time: Weight for the total-time-squared cost term.  Should
            dominate; defaults to 10.0.
        weight_deviation: Weight for the squared deviation from the
            reference path at each interior waypoint.
        weight_velocity: Weight for the velocity deviation from
            *cruise_speed*.  Prevents the degenerate solution T → 0.
        weight_collision: Weight for obstacle-penetration penalties.
        weight_dynamics: Weight for the dynamics-constraint penalty
            (speed bounds).  A large value (default 100.0) steers the
            optimiser toward feasible speed profiles.
        max_speed: Upper speed limit (world units / s).  When provided,
            implied segment speeds exceeding this value incur a penalty
            scaled by *weight_dynamics*.  Also used in the post-optimisation
            feasibility check.
        min_speed: Lower speed limit (world units / s).  When provided,
            implied segment speeds below this value incur a penalty.
        time_relaxation: Relaxation factor *α* for the initial segment
            duration estimate: ``t_i⁰ = α · L_i / cruise_speed``.  Values
            above 1.0 give the optimizer room to tighten time. Defaults to
            1.5 (50 % slack).
        method: Optimization algorithm forwarded to
            ``scipy.optimize.minimize``.  ``"L-BFGS-B"`` (default) works
            well for smooth cost landscapes; ``"SLSQP"`` is an alternative.
        sample_count: Number of intermediate points sampled *within each
            segment* for the collision term.  Higher values catch narrow
            corridors at the cost of slower evaluation.
        max_iter: Maximum number of iterations for the Stage-2 solver.
        ftol: Convergence tolerance for the Stage-2 solver (function
            value change threshold).

    Raises:
        ValueError: If *cruise_speed* is not positive.
    """

    def __init__(
        self,
        occupancy: Occupancy,
        cruise_speed: float = 1.0,
        weight_time: float = 10.0,
        weight_deviation: float = 1.0,
        weight_velocity: float = 1.0,
        weight_collision: float = 5.0,
        weight_dynamics: float = 100.0,
        max_speed: Optional[float] = None,
        min_speed: Optional[float] = None,
        time_relaxation: float = 1.5,
        method: str = "L-BFGS-B",
        sample_count: int = 3,
        max_iter: int = 500,
        ftol: float = 1e-9,
    ) -> None:
        """Initialize the TrajectoryOptimizer.

        Args:
            occupancy: Occupancy map for collision queries.
            cruise_speed: Target traversal speed (world units / s).
            weight_time: Cost weight for total-time term.
            weight_deviation: Cost weight for path-deviation term.
            weight_velocity: Cost weight for velocity-deviation term.
            weight_collision: Cost weight for collision-penalty term.
            weight_dynamics: Cost weight for dynamics-constraint penalty
                (speed bounds).
            max_speed: Upper speed limit (world units / s) for the
                dynamics penalty and post-optimisation feasibility check.
                ``None`` disables the upper-speed constraint.
            min_speed: Lower speed limit (world units / s) for the
                dynamics penalty and post-optimisation feasibility check.
                ``None`` disables the lower-speed constraint.
            time_relaxation: Relaxation factor for the initial time
                estimate.
            method: ``scipy.optimize.minimize`` method.
            sample_count: Intermediate sample count per segment for the
                collision term.
            max_iter: Maximum number of Stage-2 solver iterations.
            ftol: Stage-2 solver convergence tolerance (function value).

        Raises:
            ValueError: If *cruise_speed* is not positive.
        """
        if cruise_speed <= 0:
            raise ValueError(
                f"cruise_speed must be positive, got {cruise_speed!r}."
            )
        self.occupancy = occupancy
        self.cruise_speed = cruise_speed
        self.weight_time = weight_time
        self.weight_deviation = weight_deviation
        self.weight_velocity = weight_velocity
        self.weight_collision = weight_collision
        self.weight_dynamics = weight_dynamics
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.time_relaxation = time_relaxation
        self.method = method
        self.sample_count = sample_count
        self.max_iter = max_iter
        self.ftol = ftol

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        reference_path: List[np.ndarray],
        inverse_kinematics: Optional[
            Callable[[np.ndarray, np.ndarray, float, float], np.ndarray]
        ] = None,
        feasibility: Optional[Callable[[np.ndarray], bool]] = None,
    ) -> TrajectoryResult:
        """Optimize a trajectory from a reference path.

        Runs the two-stage optimization:

        1. **Stage 1** builds an initial guess via inverse kinematics (or
           straight-line approximation) and sets segment durations from the
           cruise speed.
        2. **Stage 2** refines the guess with
           ``scipy.optimize.minimize`` using the composite cost function.

        After optimisation, each waypoint's implied speed (segment length
        divided by segment duration) is checked against :attr:`max_speed`
        and :attr:`min_speed` when those limits are set.  If a *feasibility*
        callable is provided, it is invoked with a derived state
        ``(x, y, θ, v)`` for each waypoint, where *θ* is the segment
        heading and *v* is the implied speed.  Any violation sets
        :attr:`~TrajectoryResult.is_feasible` to ``False`` on the returned
        result.  Callers **must** check this flag and stall the vehicle
        rather than executing an infeasible trajectory.

        Args:
            reference_path: Ordered list of N+1 position arrays from the
                global planner (start + N waypoints).  Must contain at
                least two waypoints.
            inverse_kinematics: Optional callable with signature
                ``(start, goal, speed, duration) -> np.ndarray`` that
                returns control commands for a segment.  When ``None``,
                commands are estimated from the segment geometry.
            feasibility: Optional callable ``(state) -> bool`` used to
                check physical realizability of each optimized waypoint.
                The callable receives a derived state array
                ``(x, y, θ, v, ω)`` built from the waypoint position,
                inferred heading, implied segment speed, and estimated
                turn rate.  When any waypoint fails the check,
                ``result.is_feasible`` is set to ``False``.

        Returns:
            A :class:`TrajectoryResult` containing the optimized states,
            commands, segment durations, final cost, and an
            :attr:`~TrajectoryResult.is_feasible` flag.

        Raises:
            ValueError: If *reference_path* has fewer than two waypoints.
        """
        ref = [np.asarray(p, dtype=float) for p in reference_path]
        if len(ref) < 2:
            raise ValueError(
                "reference_path must contain at least two waypoints; "
                f"got {len(ref)}."
            )

        segment_count = len(ref) - 1  # N
        dim = ref[0].shape[0]

        # --- Stage 1: build initial guess ----------------------------
        x0 = self._initial_guess(ref, segment_count, dim)

        # --- Stage 2: scipy refinement -------------------------------
        bounds = self._build_bounds(segment_count, dim)
        result = minimize(
            self._cost,
            x0,
            args=(ref, segment_count, dim),
            method=self.method,
            bounds=bounds,
            options={"maxiter": self.max_iter, "ftol": self.ftol},
        )
        x_opt = result.x

        # --- Extract solution ----------------------------------------
        durations, waypoints = self._unpack(x_opt, ref, segment_count, dim)

        # Compute commands from IK (or geometry)
        commands = self._compute_commands(
            waypoints, durations, inverse_kinematics
        )

        # --- Hard feasibility check ----------------------------------
        # Build derived states (x, y, θ, v) for each waypoint so that
        # model is_feasible callables receive meaningful dynamics info.
        derived = self._compute_derived_states(waypoints, durations)
        traj_feasible = self._check_speed_bounds(durations, waypoints)
        if feasibility is not None:
            for state in derived:
                if not feasibility(state):
                    traj_feasible = False
                    break

        final_cost = float(self._cost(x_opt, ref, segment_count, dim))
        return TrajectoryResult(
            states=waypoints,
            commands=commands,
            durations=list(durations),
            cost=final_cost,
            is_feasible=traj_feasible,
        )

    # ------------------------------------------------------------------
    # Stage 1: initial guess
    # ------------------------------------------------------------------

    def _initial_guess(
        self,
        ref: List[np.ndarray],
        segment_count: int,
        dim: int,
    ) -> np.ndarray:
        """Build the Stage-1 initial guess vector.

        Segment durations are set to ``α · L_i / v_cruise``.  Interior
        waypoints start exactly on the reference path (zero deviation).

        Args:
            ref: Reference waypoints.
            segment_count: Number of segments *N*.
            dim: Spatial dimension of the waypoints.

        Returns:
            Flat initial-guess vector of length ``N + dim*(N-1)``.
        """
        # Initial segment durations
        durations = np.array(
            [
                self.time_relaxation
                * float(np.linalg.norm(ref[i + 1] - ref[i]))
                / self.cruise_speed
                for i in range(segment_count)
            ],
            dtype=float,
        )
        # Clamp: avoid zero durations on degenerate edges
        durations = np.maximum(durations, 1e-3)

        # Interior waypoints placed on the reference path
        interior_count = segment_count - 1
        if interior_count > 0:
            interior = np.concatenate(
                [ref[i + 1] for i in range(interior_count)]
            )
        else:
            interior = np.empty(0, dtype=float)

        return np.concatenate([durations, interior])

    # ------------------------------------------------------------------
    # Cost function
    # ------------------------------------------------------------------

    def _cost(
        self,
        x: np.ndarray,
        ref: List[np.ndarray],
        segment_count: int,
        dim: int,
    ) -> float:
        """Evaluate the composite cost function.

        Args:
            x: Flat decision-variable vector (durations + interior
                waypoint positions).
            ref: Reference waypoints.
            segment_count: Number of segments *N*.
            dim: Spatial dimension.

        Returns:
            Scalar composite cost.
        """
        durations, waypoints = self._unpack(x, ref, segment_count, dim)

        # Build numpy arrays for vectorised computation
        pts = np.array(waypoints)  # (N+1, dim)
        durs = np.maximum(durations, 1e-9)  # (N,)

        # --- Time cost ---------------------------------------------------
        total_time = float(np.sum(durs))
        j_time = self.weight_time * total_time**2

        # --- Velocity deviation ------------------------------------------
        diff = pts[1:] - pts[:-1]  # (N, dim)
        lengths = np.linalg.norm(diff, axis=1)  # (N,)
        speeds = lengths / durs  # (N,)
        j_velocity = self.weight_velocity * float(
            np.sum((speeds - self.cruise_speed) ** 2)
        )

        # --- Deviation from reference path (interior only) ---------------
        # Interior waypoints: indices 1 .. N-1
        interior_count = segment_count - 1
        j_deviation = 0.0
        if interior_count > 0:
            ref_interior = np.array(ref[1:-1])  # (N-1, dim)
            pts_interior = pts[1:-1]  # (N-1, dim)
            j_deviation = self.weight_deviation * float(
                np.sum((pts_interior - ref_interior) ** 2)
            )

        # --- Collision penalty (batch query) -----------------------------
        clearance = getattr(self.occupancy, "clearance", 0.5)
        j_collision = 0.0

        # Collect all query points: interior waypoints + segment samples
        query_pts_list: list[np.ndarray] = []
        if interior_count > 0:
            query_pts_list.append(pts[1:-1])

        if self.sample_count > 0:
            for i in range(segment_count):
                p_a = pts[i]
                p_b = pts[i + 1]
                alphas = np.linspace(0.0, 1.0, self.sample_count + 2)[1:-1]
                samples = p_a + alphas[:, None] * (p_b - p_a)
                query_pts_list.append(samples)

        if query_pts_list:
            all_query = np.concatenate(query_pts_list, axis=0)
            # Use batch query when available (KDTreeOccupancy)
            if hasattr(self.occupancy, "query_distances"):
                dists = self.occupancy.query_distances(all_query)
            else:
                dists = np.array(
                    [self.occupancy.nearest_obstacle(p)[0] for p in all_query]
                )
            penetrations = np.maximum(0.0, clearance - dists)
            j_collision = self.weight_collision * float(
                np.sum(penetrations**2)
            )

        # --- Dynamics penalty (speed bounds) -----------------------------
        # Penalises implied segment speeds that violate max_speed / min_speed.
        # Acts as a soft-to-hard barrier scaled by weight_dynamics.
        j_dynamics = 0.0
        if self.max_speed is not None or self.min_speed is not None:
            if self.max_speed is not None:
                over = np.maximum(0.0, speeds - self.max_speed)
                j_dynamics += float(np.sum(over**2))
            if self.min_speed is not None:
                under = np.maximum(0.0, self.min_speed - speeds)
                j_dynamics += float(np.sum(under**2))
            j_dynamics *= self.weight_dynamics

        return j_time + j_deviation + j_velocity + j_collision + j_dynamics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _unpack(
        self,
        x: np.ndarray,
        ref: List[np.ndarray],
        segment_count: int,
        dim: int,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Unpack the flat decision vector into durations and waypoints.

        Args:
            x: Flat decision-variable vector.
            ref: Reference waypoints (first and last are fixed).
            segment_count: Number of segments *N*.
            dim: Spatial dimension.

        Returns:
            ``(durations, waypoints)`` where *durations* is a length-*N*
            array and *waypoints* is a list of *N+1* position arrays
            (start and end fixed to ``ref[0]`` and ``ref[-1]``).
        """
        durations = x[:segment_count]
        interior_count = segment_count - 1
        interior_flat = x[segment_count:]

        waypoints: List[np.ndarray] = [ref[0].copy()]
        for i in range(interior_count):
            waypoints.append(interior_flat[i * dim : (i + 1) * dim].copy())
        waypoints.append(ref[-1].copy())
        return durations, waypoints

    def _build_bounds(
        self,
        segment_count: int,
        dim: int,
    ) -> List[Tuple[Optional[float], Optional[float]]]:
        """Build variable bounds for scipy.optimize.minimize.

        Duration variables have a small positive lower bound.  Position
        variables are unconstrained (the deviation cost acts as a soft
        bound).

        Args:
            segment_count: Number of segments *N*.
            dim: Spatial dimension.

        Returns:
            List of ``(lower, upper)`` tuples, one per decision variable.
        """
        duration_bounds = [(1e-3, None)] * segment_count
        position_bounds = [(None, None)] * (dim * (segment_count - 1))
        return duration_bounds + position_bounds

    def _compute_derived_states(
        self,
        waypoints: List[np.ndarray],
        durations: np.ndarray,
    ) -> List[np.ndarray]:
        """Build derived states ``(x, y, θ, v, ω)`` for each waypoint.

        The heading *θ* at each waypoint is the direction of the outgoing
        segment (or the incoming segment for the final waypoint).  The
        implied speed *v* is the segment length divided by the segment
        duration.  The turn rate *ω* is the normalized heading change
        between consecutive segments divided by the incoming segment
        duration; it is zero for the first waypoint.

        This five-element representation matches the extended state
        expected by :meth:`DubinsVehicle.is_feasible` and
        :meth:`DubinsPrimitive.is_feasible`, enabling meaningful
        dynamic-constraint checks (speed bounds and minimum turning
        radius) during post-optimisation validation.

        Args:
            waypoints: List of *N+1* optimized position arrays.
            durations: Length-*N* array of segment durations.

        Returns:
            List of *N+1* derived state arrays ``(x, y, θ, v, ω)``.
        """
        n = len(durations)

        # Pre-compute per-segment headings and speeds.
        seg_theta: List[float] = []
        seg_speed: List[float] = []
        for i in range(n):
            seg = waypoints[i + 1] - waypoints[i]
            seg_len = float(np.linalg.norm(seg))
            theta = (
                math.atan2(float(seg[1]), float(seg[0]))
                if seg_len > 1e-12
                else 0.0
            )
            seg_theta.append(theta)
            seg_speed.append(seg_len / max(float(durations[i]), 1e-9))

        derived: List[np.ndarray] = []
        for i in range(n + 1):
            pos = waypoints[i]
            # Heading and speed: outgoing segment, or incoming for last pt.
            if i < n:
                theta = seg_theta[i]
                speed = seg_speed[i]
            else:
                theta = seg_theta[n - 1]
                speed = seg_speed[n - 1]

            # Turn rate: heading change / incoming segment duration.
            if i == 0 or n < 2:
                omega = 0.0
            else:
                prev_theta = seg_theta[i - 1] if i <= n else seg_theta[n - 1]
                cur_theta = seg_theta[i] if i < n else seg_theta[n - 1]
                delta = math.atan2(
                    math.sin(cur_theta - prev_theta),
                    math.cos(cur_theta - prev_theta),
                )
                omega = delta / max(float(durations[i - 1]), 1e-9)

            x = float(pos[0]) if pos.shape[0] >= 1 else 0.0
            y = float(pos[1]) if pos.shape[0] >= 2 else 0.0
            derived.append(np.array([x, y, theta, speed, omega], dtype=float))
        return derived

    def _check_speed_bounds(
        self,
        durations: np.ndarray,
        waypoints: List[np.ndarray],
    ) -> bool:
        """Return ``True`` iff all implied segment speeds satisfy the bounds.

        A segment speed ``v_i = ||p_{i+1} - p_i|| / t_i`` is checked
        against :attr:`max_speed` and :attr:`min_speed` when those limits
        are set.

        Args:
            durations: Length-*N* array of segment durations.
            waypoints: List of *N+1* position arrays.

        Returns:
            ``True`` if all segments are within speed bounds, ``False``
            otherwise.
        """
        n = len(durations)
        for i in range(n):
            length = float(np.linalg.norm(waypoints[i + 1] - waypoints[i]))
            t_i = max(float(durations[i]), 1e-9)
            v_i = length / t_i
            if self.max_speed is not None and v_i > self.max_speed + 1e-9:
                return False
            if self.min_speed is not None and v_i < self.min_speed - 1e-9:
                return False
        return True

    def _compute_commands(
        self,
        waypoints: List[np.ndarray],
        durations: np.ndarray,
        inverse_kinematics: Optional[
            Callable[[np.ndarray, np.ndarray, float, float], np.ndarray]
        ],
    ) -> List[np.ndarray]:
        """Compute control commands for each segment.

        When *inverse_kinematics* is provided, it is called as
        ``ik(start, goal, speed, duration) -> commands``.  Otherwise a
        straight-line command estimate is used: constant speed equal to the
        segment average, and zero turn rate.

        Args:
            waypoints: List of *N+1* optimized waypoints.
            durations: Length-*N* array of segment durations.
            inverse_kinematics: Optional IK callable.

        Returns:
            List of *N* command arrays.
        """
        segment_count = len(durations)
        commands: List[np.ndarray] = []
        for i in range(segment_count):
            p_start = waypoints[i]
            p_end = waypoints[i + 1]
            t_i = float(durations[i])
            length = float(np.linalg.norm(p_end - p_start))
            speed_i = length / t_i if t_i > 1e-9 else self.cruise_speed

            if inverse_kinematics is not None:
                cmd = np.asarray(
                    inverse_kinematics(p_start, p_end, speed_i, t_i),
                    dtype=float,
                )
            else:
                # Fallback: constant speed, zero turn rate
                cmd = np.array([speed_i, 0.0], dtype=float)
            commands.append(cmd)
        return commands
