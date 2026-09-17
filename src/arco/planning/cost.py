"""Default distance and heuristic costs, re-exported from the compiled extension.

The implementation is ``CostPolicy`` in the ``arco-planning`` crate, registered back
under its Python spelling by the binding layer and reaching callers
through :mod:`arco._arco`. Subclassing it still works and still replaces the metric a planner measures with, at the cost deviation A-24 describes.
"""

from arco._arco import PlannerCost

__all__ = ["PlannerCost"]
