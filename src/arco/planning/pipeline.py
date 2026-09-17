"""PlanningPipeline, re-exported from the compiled extension.

The implementation is ``PyPlanningPipeline`` in the ``arco-py`` crate. It
sequences the three stages a caller injected, planner then pruner then
optimizer, times each of them, and records what came back in a
:class:`PipelineResult`.

Every stage is a call into the interpreter, so the loop holds the lock
throughout rather than releasing it three times for nothing. The two
serialization helpers reach ``numpy`` for the same reason the loader in
:mod:`arco.config` stays Python: ``.npz`` is numpy's format, and a second
implementation of it would be a second definition to keep in step.
"""

from arco._arco import PipelineResult, PlanningPipeline

__all__ = ["PipelineResult", "PlanningPipeline"]
