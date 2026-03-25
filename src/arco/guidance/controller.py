"""
Controllers for path tracking (pure pursuit, PID, MPC).
"""

from abc import ABC, abstractmethod


class Controller(ABC):
    """
    Abstract base for feedback controllers.
    """

    @abstractmethod
    def control(self, state, reference):
        """Compute control command to track reference from current state."""
        pass


class PurePursuitController(Controller):
    """
    Pure pursuit controller for path tracking.
    """

    def __init__(self, lookahead_distance=1.0):
        self.lookahead_distance = lookahead_distance

    def control(self, state, reference):
        # Placeholder: would compute steering command in practice
        return 0.0


class PIDController(Controller):
    """
    PID controller for path tracking.
    """

    def __init__(self, kp=1.0, ki=0.0, kd=0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def control(self, state, reference):
        # Placeholder: would compute PID output in practice
        error = reference - state
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


class MPCController(Controller):
    """
    Model Predictive Controller (stub).
    """

    def __init__(self, horizon=10, dt=0.1):
        self.horizon = horizon
        self.dt = dt

    def control(self, state, reference):
        # Placeholder: would solve optimization in practice
        return 0.0
