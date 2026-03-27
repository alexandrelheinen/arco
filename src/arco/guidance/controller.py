"""Controllers for path tracking (pure pursuit, PID, MPC)."""

from abc import ABC, abstractmethod


class Controller(ABC):
    """Abstract base for feedback controllers."""

    @abstractmethod
    def control(self, state: float, reference: float) -> float:
        """Compute control command to track reference from current state.

        Args:
            state: The current state value.
            reference: The reference/target value.
        Returns:
            Control command as a float.
        """
        pass


class PurePursuitController(Controller):
    """Pure pursuit controller for path tracking."""

    def __init__(self, lookahead_distance: float = 1.0) -> None:
        """Initialize PurePursuitController.

        Args:
            lookahead_distance: Distance ahead to look for path tracking.
        """
        self.lookahead_distance = lookahead_distance

    def control(self, state: float, reference: float) -> float:
        """Compute pure pursuit steering command (stub).

        Args:
            state: The current state value.
            reference: The reference/target value.
        Returns:
            Steering command as a float.
        """
        # Placeholder: would compute steering command in practice
        return 0.0


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
        # Placeholder: would compute PID output in practice
        error = reference - state
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


class MPCController(Controller):
    """Model Predictive Controller (stub)."""

    def __init__(self, horizon: int = 10, dt: float = 0.1) -> None:
        """Initialize MPCController.

        Args:
            horizon: Prediction horizon (number of steps).
            dt: Time step duration.
        """
        self.horizon = horizon
        self.dt = dt

    def control(self, state: float, reference: float) -> float:
        """Compute MPC control output (stub).

        Args:
            state: The current state value.
            reference: The reference/target value.
        Returns:
            Control command as a float.
        """
        # Placeholder: would compute MPC output in practice
        return 0.0
