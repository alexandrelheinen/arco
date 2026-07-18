"""MPCStepResult: diagnostics returned by one MPC tracking step."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MPCStepResult:
    """Result of a single :class:`~arco.control.mpc.base.MPCTracker` step.

    Attributes:
        speed_cmd: Commanded forward speed (m/s).
        turn_rate_cmd: Commanded turn rate (rad/s).
        cross_track_error: Signed lateral error to the reference (m).
        heading_error: Heading error wrapped to ``(−π, π]`` (rad).
        progress: Arc-length progress along the reference (m).
        predicted_clearance_min: Minimum predicted obstacle distance over
            the horizon (m).  ``inf`` when no occupancy map is set.
        solver_success: Whether the NLP solver returned a usable solution.
        solver_status: Solver status string (e.g. IPOPT return status).
        solve_time_s: Wall-clock solve time in seconds.
        cost: Optimal (or fallback) cost value.
    """

    speed_cmd: float
    turn_rate_cmd: float
    cross_track_error: float
    heading_error: float
    progress: float
    predicted_clearance_min: float
    solver_success: bool
    solver_status: str
    solve_time_s: float
    cost: float
