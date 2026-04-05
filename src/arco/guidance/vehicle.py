"""DubinsVehicle: Dubins-like unicycle kinematic model with bounded dynamics."""

from __future__ import annotations

import math

import numpy as np


class DubinsVehicle:
    """Dubins-like kinematic vehicle model with bounded dynamics.

    Implements a unicycle model with Dubins-like constraints: no-reverse
    motion (configurable ``min_speed``), bounded turn rate, and first-order
    acceleration and turn-rate filtering.

    State: ``(x, y, heading)`` in world frame.
    Controls: ``(speed, turn_rate)`` saturated and rate-limited before
    integration via forward-Euler.

    Attributes:
        x: Current x position (metres).
        y: Current y position (metres).
        heading: Current heading angle (radians), normalised to ``(−π, π]``.
        max_speed: Maximum forward speed (m/s).
        min_speed: Minimum forward speed; 0.0 prevents reversing (m/s).
        max_turn_rate: Maximum absolute turn rate (rad/s).
        max_acceleration: Maximum rate of speed change (m/s²).
        max_turn_rate_dot: Maximum rate of turn-rate change (rad/s²).
    """

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        heading: float = 0.0,
        max_speed: float = 5.0,
        min_speed: float = 0.0,
        max_turn_rate: float = 1.0,
        max_acceleration: float = 2.0,
        max_turn_rate_dot: float = 2.0,
    ) -> None:
        """Initialize DubinsVehicle.

        Args:
            x: Initial x position in world frame (metres).
            y: Initial y position in world frame (metres).
            heading: Initial heading angle (radians).
            max_speed: Maximum forward speed (m/s).
            min_speed: Minimum forward speed; set to 0.0 to prevent reversing,
                negative to allow reversing (m/s).
            max_turn_rate: Maximum absolute turn rate (rad/s).
            max_acceleration: Maximum rate of speed change (m/s²).
            max_turn_rate_dot: Maximum rate of turn-rate change (rad/s²).
        """
        self.x = x
        self.y = y
        self.heading = heading
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.max_turn_rate = max_turn_rate
        self.max_acceleration = max_acceleration
        self.max_turn_rate_dot = max_turn_rate_dot

        self._speed: float = 0.0
        self._turn_rate: float = 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def pose(self) -> tuple[float, float, float]:
        """Current pose as ``(x, y, heading)``."""
        return (self.x, self.y, self.heading)

    @property
    def speed(self) -> float:
        """Current forward speed (m/s)."""
        return self._speed

    @property
    def turn_rate(self) -> float:
        """Current turn rate (rad/s)."""
        return self._turn_rate

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(
        self, x: float = 0.0, y: float = 0.0, heading: float = 0.0
    ) -> None:
        """Reset vehicle state to a new pose with zero speed and turn rate.

        Args:
            x: New x position (metres).
            y: New y position (metres).
            heading: New heading angle (radians).
        """
        self.x = x
        self.y = y
        self.heading = heading
        self._speed = 0.0
        self._turn_rate = 0.0

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def step(
        self, speed_cmd: float, turn_rate_cmd: float, dt: float
    ) -> tuple[float, float, float]:
        """Integrate kinematics one step with command saturation and filtering.

        Applies acceleration filtering to speed and turn-rate-dot filtering to
        turn rate, then integrates the unicycle kinematic equations with a
        forward-Euler step.

        Args:
            speed_cmd: Desired speed command (m/s).
            turn_rate_cmd: Desired turn rate command (rad/s).
            dt: Time step duration (s).

        Returns:
            Updated pose as ``(x, y, heading)``.
        """
        # --- Acceleration filtering: rate-limit speed change ---
        max_delta_speed = self.max_acceleration * dt
        delta_speed = float(
            np.clip(speed_cmd - self._speed, -max_delta_speed, max_delta_speed)
        )
        self._speed = float(
            np.clip(self._speed + delta_speed, self.min_speed, self.max_speed)
        )

        # --- Turn-rate-dot filtering: rate-limit turn rate change ---
        max_delta_turn = self.max_turn_rate_dot * dt
        delta_turn = float(
            np.clip(
                turn_rate_cmd - self._turn_rate,
                -max_delta_turn,
                max_delta_turn,
            )
        )
        self._turn_rate = float(
            np.clip(
                self._turn_rate + delta_turn,
                -self.max_turn_rate,
                self.max_turn_rate,
            )
        )

        # --- Forward-Euler integration of unicycle kinematics ---
        self.x += self._speed * math.cos(self.heading) * dt
        self.y += self._speed * math.sin(self.heading) * dt
        self.heading += self._turn_rate * dt

        # Normalise heading to (−π, π]
        self.heading = math.atan2(
            math.sin(self.heading), math.cos(self.heading)
        )

        return self.pose

    # ------------------------------------------------------------------
    # Trajectory optimiser interface
    # ------------------------------------------------------------------

    def inverse_kinematics(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        speed: float,
        duration: float,
    ) -> np.ndarray:
        """Compute control commands to steer from *start* toward *goal*.

        This naive unicycle inverse-kinematics sets the speed to *speed*
        and derives the required turn rate from the heading difference
        between the current heading (inferred from *start*) and the
        direction toward *goal*.  When *start* has three or more elements
        the third component is taken as the current heading; otherwise a
        zero heading is assumed.

        The result is an admissible initial guess for the
        :class:`~arco.planning.continuous.optimizer.TrajectoryOptimizer`
        Stage-1 initialisation.  Saturates turn rate to
        :attr:`max_turn_rate`.

        Args:
            start: Starting position ``(x, y)`` or state ``(x, y, θ, …)``.
            goal: Target position ``(x, y)`` or state ``(x, y, θ, …)``.
            speed: Desired traversal speed (m/s).
            duration: Time budget for the segment (s).  Must be positive.

        Returns:
            Command vector ``(speed_cmd, turn_rate_cmd)`` as a numpy
            array of shape ``(2,)``.
        """
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)

        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        target_heading = math.atan2(dy, dx)

        current_heading = float(start[2]) if start.shape[0] >= 3 else 0.0

        # Shortest-path heading difference in (−π, π]
        heading_diff = math.atan2(
            math.sin(target_heading - current_heading),
            math.cos(target_heading - current_heading),
        )

        safe_dur = max(duration, 1e-9)
        turn_rate = float(
            np.clip(
                heading_diff / safe_dur,
                -self.max_turn_rate,
                self.max_turn_rate,
            )
        )
        speed_cmd = float(np.clip(speed, self.min_speed, self.max_speed))
        return np.array([speed_cmd, turn_rate], dtype=float)

    def is_feasible(self, state: np.ndarray) -> bool:
        """Check whether a state is within the vehicle's dynamic limits.

        Accepts a 3-element kinematic state ``(x, y, θ)`` — always
        feasible — or a 5-element extended state
        ``(x, y, θ, speed, turn_rate)`` which is checked against
        :attr:`max_speed`, :attr:`min_speed`, and :attr:`max_turn_rate`.

        Args:
            state: Kinematic state ``(x, y, θ)`` or extended state
                ``(x, y, θ, speed, turn_rate)``.

        Returns:
            ``True`` if the state satisfies all dynamic constraints,
            ``False`` otherwise.
        """
        state = np.asarray(state, dtype=float)
        if state.shape[0] >= 5:
            speed = float(state[3])
            turn_rate = float(state[4])
            if speed < self.min_speed or speed > self.max_speed:
                return False
            if abs(turn_rate) > self.max_turn_rate:
                return False
        elif state.shape[0] >= 4:
            speed = float(state[3])
            if speed < self.min_speed or speed > self.max_speed:
                return False
        return True
