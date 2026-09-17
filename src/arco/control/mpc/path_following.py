"""Path-following MPCC: the compiled controller and its configuration.

The controller is ``PathFollowingMpc`` in the ``arco-control`` crate,
reached through the binding layer. ADR-002: the nonlinear program becomes
a short sequence of convex programs, so the commands differ from the ones
the Python produced. Deviation A-30 covers the obstacle barrier, which
leaves ``obstacle_barrier_power`` inert, A-31 the reported cost and A-32
the solver status strings.

The two records stay dataclasses here. They carry no algorithm, they load
from the packaged YAML through :mod:`arco.config`, and callers build
variants of them with :func:`dataclasses.replace`, which only works on a
real dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from arco._arco import DubinsPathFollowingMPC
from arco.config import load_config


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
        max_solver_iter_count: Interior-point iteration budget.
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


__all__ = [
    "DubinsPathFollowingMPC",
    "DubinsVehicleLimits",
    "PathFollowingMPCConfig",
]
