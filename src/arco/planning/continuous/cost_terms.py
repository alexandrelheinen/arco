"""Default CostTerm helpers, re-exported from the compiled extension.

``TimeCostTerm``, ``DeviationCostTerm``, ``VelocityCostTerm``,
``CollisionCostTerm`` and ``DynamicsCostTerm`` together reproduce the
historical five-term composite cost (time, deviation, velocity,
collision plus barrier, dynamics), and ``build_default_cost_terms``
builds one of each in that order. The implementations live in the
``arco-planning`` and ``arco-py`` crates, registered back under their
Python spellings by the binding layer and reaching callers through
:mod:`arco._arco`. A caller's own object implementing
:class:`~arco.protocols.CostTerm` still works anywhere one of these
does, since ``TrajectoryOptimizer`` accepts a ``cost_terms=`` list of
any object carrying a ``name`` attribute and a ``__call__`` method.
"""

from arco._arco import (
    CollisionCostTerm,
    DeviationCostTerm,
    DynamicsCostTerm,
    TimeCostTerm,
    VelocityCostTerm,
    build_default_cost_terms,
)

__all__ = [
    "CollisionCostTerm",
    "DeviationCostTerm",
    "DynamicsCostTerm",
    "TimeCostTerm",
    "VelocityCostTerm",
    "build_default_cost_terms",
]
