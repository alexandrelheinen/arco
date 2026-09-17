"""Path shortening, re-exported from the compiled extension.

The implementation is ``TrajectoryPruner`` in the ``arco-planning`` crate, registered back
under its Python spelling by the binding layer and reaching callers
through :mod:`arco._arco`.
"""

from arco._arco import TrajectoryPruner

__all__ = ["TrajectoryPruner"]
