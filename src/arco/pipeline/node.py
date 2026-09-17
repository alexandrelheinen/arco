"""PipelineNode, re-exported from the compiled extension.

The implementation is ``PipelineNode`` in the ``arco-runtime`` crate. A
subclass overrides :meth:`run`, which the node calls once on its own
background thread, and publishes frames onto the shared bus from there.
"""

from arco._arco import PipelineNode

__all__ = ["PipelineNode"]
