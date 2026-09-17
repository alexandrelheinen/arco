"""Discrete planner base, re-exported from the compiled extension.

The implementation is the ``DiscreteMap`` trait in the ``arco-core`` crate, registered back
under its Python spelling by the binding layer and reaching callers
through :mod:`arco._arco`.
"""

from arco._arco import DiscretePlanner

__all__ = ["DiscretePlanner"]
