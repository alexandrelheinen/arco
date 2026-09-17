"""Soft-barrier cost helpers, re-exported from the compiled extension.

The implementations are ``obstacle_barrier`` and ``forward_cone_factor``
in the ``arco-control`` crate. Both take and return floats. The symbolic
branch that accepted CasADi expressions is gone with CasADi itself
(ADR-002), because there is no longer a nonlinear graph to build.
"""

from arco._arco import forward_cone_factor, obstacle_barrier

__all__ = ["forward_cone_factor", "obstacle_barrier"]
