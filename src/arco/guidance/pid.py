"""PIDController: proportional-integral-derivative feedback controller."""

from __future__ import annotations

from .controller import Controller


class PIDController(Controller):
    """PID controller for path tracking."""

    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.1) -> None:
        """Initialize PIDController.

        Args:
            kp: Proportional gain.
            ki: Integral gain.
            kd: Derivative gain.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def control(self, state: float, reference: float) -> float:
        """Compute PID control output.

        Args:
            state: The current state value.
            reference: The reference/target value.

        Returns:
            Control command as a float.
        """
        error = reference - state
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative
