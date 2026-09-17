"""PipelineRunner, re-exported from the compiled extension.

The implementation is ``PipelineRunner`` in the ``arco-runtime`` crate. It
owns the shared bus, starts and stops the registered nodes, and attaches
late subscribers to a frame type.
"""

from arco._arco import PipelineRunner

__all__ = ["PipelineRunner"]
