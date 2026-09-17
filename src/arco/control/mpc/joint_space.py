"""Joint-space MPC: the compiled controller and its configuration record.

The controller is ``JointSpaceMpc`` in the ``arco-control`` crate, reached
through the binding layer. ADR-002: the nonlinear program CasADi built and
IPOPT solved is now one convex program per step, solved by Clarabel, so
the commands differ from the ones the Python produced. Deviation A-30
covers the obstacle barrier, which leaves ``obstacle_barrier_power``
inert, and A-33 covers what braking does when no answer comes back.

The configuration stays a dataclass here. It carries no algorithm, it
loads from the packaged YAML through :mod:`arco.config`, and callers
build variants of it with :func:`dataclasses.replace`, which only works
on a real dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from arco._arco import JointSpaceMPC
from arco.config import load_config


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

    def with_horizon_overrides(
        self,
        *,
        step_count: int | None = None,
        dt: float | None = None,
    ) -> JointSpaceMPCConfig:
        """Return a copy with optional horizon overrides applied.

        Args:
            step_count: Optional new prediction horizon length (steps).
            dt: Optional new discretization step (s).

        Returns:
            A new :class:`JointSpaceMPCConfig` with the requested horizon
            fields replaced; other fields are unchanged.
        """
        return replace(
            self,
            horizon_step_count=(
                self.horizon_step_count
                if step_count is None
                else int(step_count)
            ),
            dt=self.dt if dt is None else float(dt),
        )


__all__ = ["JointSpaceMPC", "JointSpaceMPCConfig"]
