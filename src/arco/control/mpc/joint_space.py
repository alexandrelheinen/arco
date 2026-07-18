"""JointSpaceMPC: N-DOF carrot-tracking NMPC with soft C-space barriers.

Drop-in parallel to :class:`~arco.control.joint_tracker.JointSpaceTracker`
(same ``reset`` / ``step`` surface) that replaces proportional control +
APF repulsion with a short-horizon CasADi/IPOPT solve.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from arco.config import load_config

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
            "CasADi is required for JointSpaceMPC. "
            "Install with: pip install arco[mpc]"
        ) from exc
    return ca


@dataclass
class JointSpaceMPCConfig:
    """Horizon and weights for joint-space carrot-tracking MPC.

    Attributes:
        horizon_step_count: Prediction horizon length (steps).
        dt: Discretization step (s).
        weight_tracking: Configuration tracking weight.
        weight_velocity: Velocity regularization weight.
        weight_control: Acceleration effort weight.
        weight_obstacle: Soft obstacle-barrier weight.
        obstacle_barrier_power: Barrier exponent.
        max_solver_iter_count: IPOPT iteration budget.
    """

    horizon_step_count: int = 12
    dt: float = 0.05
    weight_tracking: float = 20.0
    weight_velocity: float = 0.5
    weight_control: float = 0.05
    weight_obstacle: float = 60.0
    obstacle_barrier_power: float = 4.0
    max_solver_iter_count: int = 40

    @staticmethod
    def create_from_config() -> JointSpaceMPCConfig:
        """Load defaults from ``config/mpc.yml`` (``joint_space`` section).

        Returns:
            Configured :class:`JointSpaceMPCConfig`.
        """
        cfg = load_config("mpc")
        js = cfg.get("joint_space", {})
        horizon = js.get("horizon", cfg.get("horizon", {}))
        weights = js.get("weights", {})
        barrier = js.get("obstacle_barrier", cfg.get("obstacle_barrier", {}))
        solver = js.get("solver", cfg.get("solver", {}))
        return JointSpaceMPCConfig(
            horizon_step_count=int(horizon.get("step_count", 12)),
            dt=float(horizon.get("dt", 0.05)),
            weight_tracking=float(weights.get("tracking", 20.0)),
            weight_velocity=float(weights.get("velocity", 0.5)),
            weight_control=float(weights.get("control", 0.05)),
            weight_obstacle=float(weights.get("obstacle", 60.0)),
            obstacle_barrier_power=float(barrier.get("power", 4.0)),
            max_solver_iter_count=int(solver.get("max_iter_count", 40)),
        )


class JointSpaceMPC:
    """N-DOF receding-horizon tracker for C-space carrots.

    API-compatible with :class:`~arco.control.joint_tracker.JointSpaceTracker`:
    ``reset(q0)`` then ``step(target_q, dt) -> q``.  Obstacle avoidance is
    inside the optimizer (no APF blend).  The unused *repulsion_gain* /
    *proportional_gain* kwargs are accepted for drop-in call-site parity.
    """

    _OBSTACLE_SAMPLE_COUNT = 4

    def __init__(
        self,
        max_vel: float | np.ndarray,
        max_acc: float | np.ndarray,
        proportional_gain: float = 2.0,
        occupancy: Occupancy | None = None,
        repulsion_gain: float = 0.0,
        config: Optional[JointSpaceMPCConfig] = None,
    ) -> None:
        """Initialize JointSpaceMPC.

        Args:
            max_vel: Per-axis velocity limit (scalar or 1-D array).
            max_acc: Per-axis acceleration limit (scalar or 1-D array).
            proportional_gain: Accepted for API parity; unused by the NLP.
            occupancy: Optional C-space occupancy for soft barriers.
            repulsion_gain: Accepted for API parity; unused (barriers are
                inside the optimizer).
            config: Optional MPC weights/horizon.  Defaults load from
                ``mpc.yml`` when available, else dataclass defaults.

        Raises:
            ImportError: If CasADi is missing.
            ValueError: If velocity/acceleration limits are not positive.
        """
        del proportional_gain, repulsion_gain
        self._ca = _require_casadi()
        self._max_vel = np.atleast_1d(np.asarray(max_vel, dtype=float))
        self._max_acc = np.atleast_1d(np.asarray(max_acc, dtype=float))
        if np.any(self._max_vel <= 0.0):
            raise ValueError(
                f"max_vel must be strictly positive; got {max_vel!r}."
            )
        if np.any(self._max_acc <= 0.0):
            raise ValueError(
                f"max_acc must be strictly positive; got {max_acc!r}."
            )
        self._dof = int(self._max_vel.shape[0])
        if self._max_acc.shape[0] == 1 and self._dof > 1:
            self._max_acc = np.full(self._dof, float(self._max_acc[0]))
        if self._max_vel.shape[0] == 1 and self._dof > 1:
            self._max_vel = np.full(self._dof, float(self._max_vel[0]))
        self._occ = occupancy
        self.config = config or JointSpaceMPCConfig()
        self.q: np.ndarray = np.zeros(self._dof)
        self.vel: np.ndarray = np.zeros(self._dof)
        self._nlp: dict[str, Any] | None = None
        self._warm_q: np.ndarray | None = None
        self._warm_v: np.ndarray | None = None
        self._warm_a: np.ndarray | None = None
        self.last_solver_success: bool = True
        self.last_solve_time_s: float = 0.0

    def reset(self, q0: np.ndarray) -> None:
        """Reset tracker state to initial configuration *q0*.

        Args:
            q0: Initial configuration array.
        """
        self.q = np.asarray(q0, dtype=float).copy()
        self.vel = np.zeros_like(self.q)
        self._warm_q = None
        self._warm_v = None
        self._warm_a = None

    def step(self, target_q: np.ndarray, dt: float) -> np.ndarray:
        """Run one NMPC step toward *target_q* and return the new configuration.

        Args:
            target_q: Carrot configuration on the planned path.
            dt: Integration time step (seconds).

        Returns:
            Updated configuration after applying the first optimal command.
        """
        target = np.asarray(target_q, dtype=float).reshape(-1)
        if target.shape[0] != self._dof:
            raise ValueError(
                f"target_q length {target.shape[0]} != DOF {self._dof}."
            )
        if not np.all(np.isfinite(self.q)) or not np.all(np.isfinite(target)):
            return self._safe_decelerate(dt)

        obstacles = self._collect_obstacles()
        clearance = float(getattr(self._occ, "clearance", 1.0) or 1.0)
        slot_count = (
            self._OBSTACLE_SAMPLE_COUNT if self._occ is not None else 0
        )
        while len(obstacles) < slot_count:
            obstacles.append(np.full(self._dof, 1e6))
        obstacles = obstacles[:slot_count]

        t0 = time.perf_counter()
        try:
            sol = self._solve(target, obstacles, clearance, slot_count)
            self.last_solve_time_s = time.perf_counter() - t0
        except Exception:  # noqa: BLE001
            self.last_solve_time_s = time.perf_counter() - t0
            self.last_solver_success = False
            return self._safe_decelerate(dt)

        if sol is None:
            self.last_solver_success = False
            return self._safe_decelerate(dt)

        self.last_solver_success = True
        self.q = sol["q1"]
        self.vel = sol["v1"]
        self._warm_q = sol["Q"]
        self._warm_v = sol["V"]
        self._warm_a = sol["A"]
        return self.q.copy()

    def _safe_decelerate(self, dt: float) -> np.ndarray:
        """Brake with maximum deceleration; keep configuration continuous."""
        acc = -np.sign(self.vel) * self._max_acc
        # Zero velocity axes stay at zero.
        acc = np.where(np.abs(self.vel) < 1e-9, 0.0, acc)
        self.vel = np.clip(self.vel + acc * dt, -self._max_vel, self._max_vel)
        self.q = self.q + self.vel * dt
        return self.q.copy()

    def _collect_obstacles(self) -> list[np.ndarray]:
        if self._occ is None or not hasattr(self._occ, "nearest_obstacle"):
            return []
        samples: list[np.ndarray] = []
        pts = [self.q.copy()]
        # Probe a short distance along the current velocity and toward
        # a few offsets for a richer barrier set.
        if np.linalg.norm(self.vel) > 1e-6:
            pts.append(self.q + self.vel * self.config.dt * 4.0)
        seen: set[tuple[float, ...]] = set()
        for pt in pts:
            _dist, nearest = self._occ.nearest_obstacle(pt)
            key = tuple(np.round(np.asarray(nearest, dtype=float), 4))
            if key in seen:
                continue
            seen.add(key)
            samples.append(np.asarray(nearest, dtype=float).copy())
        return samples

    def _build_nlp(self, obstacle_count: int) -> dict[str, Any]:
        ca = self._ca
        cfg = self.config
        n = cfg.horizon_step_count
        dt = cfg.dt
        dof = self._dof

        opti = ca.Opti()
        Q = opti.variable(dof, n + 1)
        V = opti.variable(dof, n + 1)
        A = opti.variable(dof, n)

        q0 = opti.parameter(dof)
        v0 = opti.parameter(dof)
        target = opti.parameter(dof)
        clearance_p = opti.parameter(1)
        obs = (
            opti.parameter(dof, obstacle_count) if obstacle_count > 0 else None
        )

        opti.subject_to(Q[:, 0] == q0)
        opti.subject_to(V[:, 0] == v0)

        cost = 0
        for k in range(n):
            q_k = Q[:, k]
            v_k = V[:, k]
            a_k = A[:, k]
            cost += cfg.weight_tracking * ca.sumsqr(q_k - target)
            cost += cfg.weight_velocity * ca.sumsqr(v_k)
            cost += cfg.weight_control * ca.sumsqr(a_k)

            if obs is not None:
                for j in range(obstacle_count):
                    diff = q_k - obs[:, j]
                    dist = ca.sqrt(ca.sumsqr(diff) + 1e-9)
                    penetration = (clearance_p - dist) / ca.fmax(
                        clearance_p, 1e-6
                    )
                    penetration = ca.fmax(0.0, penetration)
                    cost += cfg.weight_obstacle * (
                        penetration**cfg.obstacle_barrier_power
                    )

            v_next = v_k + a_k * dt
            opti.subject_to(Q[:, k + 1] == q_k + v_k * dt)
            opti.subject_to(V[:, k + 1] == v_next)
            for i in range(dof):
                opti.subject_to(
                    opti.bounded(
                        -float(self._max_vel[i]),
                        v_next[i],
                        float(self._max_vel[i]),
                    )
                )
                opti.subject_to(
                    opti.bounded(
                        -float(self._max_acc[i]),
                        a_k[i],
                        float(self._max_acc[i]),
                    )
                )

        cost += cfg.weight_tracking * 2.0 * ca.sumsqr(Q[:, n] - target)
        opti.minimize(cost)
        opti.solver(
            "ipopt",
            {
                "ipopt.print_level": 0,
                "print_time": 0,
                "ipopt.max_iter": int(cfg.max_solver_iter_count),
                "ipopt.sb": "yes",
                "ipopt.tol": 1e-4,
                "ipopt.warm_start_init_point": "yes",
            },
        )
        return {
            "opti": opti,
            "Q": Q,
            "V": V,
            "A": A,
            "q0": q0,
            "v0": v0,
            "target": target,
            "clearance": clearance_p,
            "obs": obs,
            "obstacle_count": obstacle_count,
        }

    def _ensure_nlp(self, obstacle_count: int) -> dict[str, Any]:
        if (
            self._nlp is None
            or int(self._nlp["obstacle_count"]) != obstacle_count
        ):
            self._nlp = self._build_nlp(obstacle_count)
        return self._nlp

    def _solve(
        self,
        target: np.ndarray,
        obstacles: list[np.ndarray],
        clearance: float,
        slot_count: int,
    ) -> dict[str, Any] | None:
        cfg = self.config
        n = cfg.horizon_step_count
        dof = self._dof
        nlp = self._ensure_nlp(slot_count)
        opti = nlp["opti"]

        opti.set_value(nlp["q0"], self.q)
        opti.set_value(nlp["v0"], self.vel)
        opti.set_value(nlp["target"], target)
        opti.set_value(nlp["clearance"], max(clearance, 1e-3))
        if slot_count > 0 and nlp["obs"] is not None:
            obs_mat = np.column_stack(obstacles)
            opti.set_value(nlp["obs"], obs_mat)

        if self._warm_q is not None and self._warm_q.shape == (dof, n + 1):
            Q_init = np.hstack([self._warm_q[:, 1:], self._warm_q[:, -1:]])
            V_init = np.hstack([self._warm_v[:, 1:], self._warm_v[:, -1:]])
            A_init = np.hstack([self._warm_a[:, 1:], self._warm_a[:, -1:]])
            Q_init[:, 0] = self.q
            V_init[:, 0] = self.vel
        else:
            Q_init = np.zeros((dof, n + 1))
            V_init = np.zeros((dof, n + 1))
            A_init = np.zeros((dof, n))
            Q_init[:, 0] = self.q
            V_init[:, 0] = self.vel
            for k in range(n):
                alpha = (k + 1) / n
                Q_init[:, k + 1] = (1.0 - alpha) * self.q + alpha * target
                V_init[:, k + 1] = np.clip(
                    (target - self.q) / max(n * cfg.dt, 1e-3),
                    -self._max_vel,
                    self._max_vel,
                )

        opti.set_initial(nlp["Q"], Q_init)
        opti.set_initial(nlp["V"], V_init)
        opti.set_initial(nlp["A"], A_init)

        try:
            sol = opti.solve()
        except RuntimeError:
            return None

        Q_opt = np.array(sol.value(nlp["Q"]), dtype=float).reshape(dof, n + 1)
        V_opt = np.array(sol.value(nlp["V"]), dtype=float).reshape(dof, n + 1)
        A_opt = np.array(sol.value(nlp["A"]), dtype=float).reshape(dof, n)
        return {
            "q1": Q_opt[:, 1].copy(),
            "v1": np.clip(V_opt[:, 1], -self._max_vel, self._max_vel),
            "Q": Q_opt,
            "V": V_opt,
            "A": A_opt,
        }
