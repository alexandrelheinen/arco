"""Async pipeline orchestration for the ARCO processing chain.

This package provides the infrastructure for running the mapping,
planning, and guidance pipeline stages as independent threads that
communicate through a shared in-memory bus.

Public API:
    - :class:`~arco.pipeline.node.PipelineNode` — abstract base class
      for a single pipeline stage (thread lifecycle + bus publishing).
    - :class:`~arco.pipeline.runner.PipelineRunner` — orchestrates
      registered nodes, owns the shared bus, and supports late
      subscriber attachment.
"""

from .node import PipelineNode
from .runner import PipelineRunner

__all__ = ["PipelineNode", "PipelineRunner"]
