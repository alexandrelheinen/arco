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
