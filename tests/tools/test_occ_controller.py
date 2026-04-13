"""Integration tests: OCC PD controller step response.

Verifies that the position gains produce ~10 % overshoot and that the
body settles at the goal, matching the design intent documented in
``src/arco/tools/config/map/occ.yml``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from arco.control.actuator import ActuatorArray
from arco.control.rigid_body import SquareBody

# ---------------------------------------------------------------------------
# Reference gains (must match occ.yml / _compute_desired_wrench defaults)
# ---------------------------------------------------------------------------

_CTRL = {
    "kp_pos": 2.5,
    "kd_pos": 4.2,
    "kp_psi": 0.50,
    "kd_psi": 0.55,
}

_DT = 0.02  # seconds, matches simulator.dt


# ---------------------------------------------------------------------------
# Helper: replicate _compute_desired_wrench without importing main/occ.py
# ---------------------------------------------------------------------------


def _wrap(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _wrench(body: SquareBody, goal: np.ndarray, cfg: dict) -> np.ndarray:
    pose = body.pose
    vel = body.velocity
    ex = float(goal[0]) - float(pose[0])
    ey = float(goal[1]) - float(pose[1])
    e_psi = _wrap(float(goal[2]) - float(pose[2]))
    fx = cfg["kp_pos"] * ex - cfg["kd_pos"] * float(vel[0])
    fy = cfg["kp_pos"] * ey - cfg["kd_pos"] * float(vel[1])
    tau = cfg["kp_psi"] * e_psi - cfg["kd_psi"] * float(vel[2])
    return np.array([fx, fy, tau])


def _simulate(
    goal: np.ndarray,
    cfg: dict,
    n_steps: int,
    dt: float = _DT,
) -> tuple[SquareBody, list[float]]:
    """Simulate the closed-loop position step response.

    Returns:
        Tuple of (final body, trajectory of x-positions).
    """
    body = SquareBody(mass=5.0, side_length=0.5)
    acts = ActuatorArray(actuator_count=3, standoff=0.05)
    xs: list[float] = []
    for _ in range(n_steps):
        w = _wrench(body, goal, cfg)
        acts.update_angles_for_target(body, w)
        f = acts.allocate_forces(w, body)
        acts.apply_to_body(f, body)
        body.step(dt)
        xs.append(float(body.pose[0]))
    return body, xs


# ---------------------------------------------------------------------------
# Design parameters (for documentation / assertion context)
# ---------------------------------------------------------------------------
# SquareBody: m=5 kg, I = m·L²/6 = 5·0.25/6 ≈ 0.208 kg·m²
# Position: ωn = sqrt(kp/m) = sqrt(2.5/5) ≈ 0.707 rad/s
#           ζ  = kd/(2·sqrt(kp·m)) = 4.2/(2·sqrt(12.5)) ≈ 0.594
#           OS% = exp(-π·ζ/sqrt(1-ζ²)) ≈ 9.8 %
# Settling time (2 %): ~11 s  → 550 steps @ dt=0.02
# Peak time (overshoot): tp = π/ωd ≈ 5.5 s → 275 steps
# ---------------------------------------------------------------------------


class TestPositionStepResponse:
    def test_overshoot_approximately_ten_percent(self) -> None:
        """Position step response: overshoot must be in [5 %, 20 %]."""
        goal = np.array([1.0, 0.0, 0.0])
        _, xs = _simulate(goal, _CTRL, n_steps=1000)  # 20 s, well past peak
        peak_x = max(xs)
        overshoot = peak_x / float(goal[0]) - 1.0
        assert 0.05 <= overshoot <= 0.20, (
            f"Expected overshoot in [5 %, 20 %], got {overshoot * 100:.1f} %"
        )

    def test_settles_within_two_percent(self) -> None:
        """Body settles to within 2 % of goal after 30 s (1500 steps)."""
        goal = np.array([1.0, 0.0, 0.0])
        body, _ = _simulate(goal, _CTRL, n_steps=1500)
        assert abs(body.pose[0] - float(goal[0])) < 0.02

    def test_peak_body_speed_within_race_speed(self) -> None:
        """Peak body speed ≤ race_speed (0.3 m/s) for a typical waypoint step.

        The planner produces waypoints spaced at step_size = 0.20 m.  For
        this inter-waypoint step the transient peak speed must not exceed
        the design race_speed.
        """
        race_speed = 0.3  # m/s — from occ.yml
        step_size = 0.20  # m  — from occ.yml planner.step_size
        body = SquareBody(mass=5.0, side_length=0.5)
        acts = ActuatorArray(actuator_count=3, standoff=0.05)
        goal = np.array([step_size, 0.0, 0.0])
        peak_v = 0.0
        for _ in range(500):
            w = _wrench(body, goal, _CTRL)
            acts.update_angles_for_target(body, w)
            f = acts.allocate_forces(w, body)
            acts.apply_to_body(f, body)
            body.step(_DT)
            v = math.hypot(float(body.velocity[0]), float(body.velocity[1]))
            peak_v = max(peak_v, v)
        assert peak_v <= race_speed, (
            f"Peak speed {peak_v:.3f} m/s exceeds race_speed {race_speed} m/s "
            f"for a step_size={step_size} m waypoint"
        )


class TestHeadingStepResponse:
    def test_heading_settles_without_large_overshoot(self) -> None:
        """Heading step response: overshoot < 10 % (near-critical damping)."""
        goal = np.array([0.0, 0.0, math.pi / 2])
        body = SquareBody(mass=5.0, side_length=0.5)
        acts = ActuatorArray(actuator_count=3, standoff=0.05)
        peak_psi = 0.0
        for _ in range(1000):
            w = _wrench(body, goal, _CTRL)
            acts.update_angles_for_target(body, w)
            f = acts.allocate_forces(w, body)
            acts.apply_to_body(f, body)
            body.step(_DT)
            peak_psi = max(peak_psi, float(body.pose[2]))
        overshoot = peak_psi / float(goal[2]) - 1.0
        assert overshoot < 0.10, (
            f"Heading overshoot {overshoot * 100:.1f} % should be < 10 %"
        )

    def test_heading_settles(self) -> None:
        """Body heading reaches goal within 0.05 rad after 30 s."""
        goal = np.array([0.0, 0.0, math.pi / 4])
        body, _ = _simulate(goal, _CTRL, n_steps=1500)
        assert abs(body.pose[2] - float(goal[2])) < 0.05
