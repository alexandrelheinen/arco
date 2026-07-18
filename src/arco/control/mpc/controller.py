"""MPCController: deprecated scalar MPC stub."""

from __future__ import annotations

import warnings

from arco.control.base import Controller


class MPCController(Controller):
    """Deprecated scalar Model Predictive Controller stub.

    .. deprecated:: 0.4.0
        Use :class:`~arco.control.mpc.path_following.DubinsPathFollowingMPC`
        for multi-state path-following MPC instead.
    """

    def __init__(self, horizon: int = 10, dt: float = 0.1) -> None:
        """Initialize MPCController.

        Args:
            horizon: Prediction horizon (number of steps).
            dt: Time step duration in seconds.
        """
        warnings.warn(
            "MPCController is deprecated; use DubinsPathFollowingMPC "
            "from arco.control.mpc for path-following MPC.",
            DeprecationWarning,
            stacklevel=2,
        )
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
        return 0.0
