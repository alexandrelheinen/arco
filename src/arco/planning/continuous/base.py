"""Continuous planner base, re-exported from the compiled extension.

The implementation is ``CostPolicy`` in the ``arco-planning`` crate, registered back
under its Python spelling by the binding layer and reaching callers
through :mod:`arco._arco`.
"""

from arco._arco import ContinuousPlanner

__all__ = ["ContinuousPlanner"]
