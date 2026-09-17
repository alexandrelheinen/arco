"""Trajectory optimization, re-exported from the compiled extension.

The implementation is ``TrajectoryOptimizer`` in the ``arco-planning`` crate, registered back
under its Python spelling by the binding layer and reaching callers
through :mod:`arco._arco`. The solver underneath is ``argmin`` rather than ``scipy``, per deviation A-08, so the cost achieved is comparable and the solution vector is not.
"""

from arco._arco import TrajectoryOptimizer, TrajectoryResult

__all__ = ["TrajectoryOptimizer", "TrajectoryResult"]
