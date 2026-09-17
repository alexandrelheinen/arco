"""MPCStepResult, re-exported from the compiled extension.

The implementation is ``PyMpcStepResult`` in the ``arco-py`` crate,
registered back under its Python spelling by the binding layer. Deviation
A-31: ``cost`` reports the surrogate convex objective the port solves, so
it is comparable across steps of one controller and not against a number
the CasADi implementation printed. Deviation A-32: ``solver_status``
carries the convex solver's vocabulary rather than IPOPT's.
"""

from arco._arco import MPCStepResult

__all__ = ["MPCStepResult"]
