"""Controller: abstract base for feedback controllers."""

from abc import ABC, abstractmethod


class Controller(ABC):
    """Abstract base for feedback controllers.

    Subclasses interpret ``state`` and ``reference`` according to their
    specific control law (e.g., path position for pure pursuit, error
    signal for PID, predicted trajectory for MPC).
    """

    @abstractmethod
    def control(self, state: float, reference: float) -> float:
        """Compute control command to track reference from current state.

        Args:
            state: The current state value (interpretation depends on
                the concrete controller subclass).
            reference: The reference/target value to track.

        Returns:
            Control command as a float.
        """
        pass
