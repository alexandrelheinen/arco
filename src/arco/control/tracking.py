"""Closed-loop path tracking, re-exported from the compiled extension.

The implementation is ``TrackingLoop`` in the ``arco-control`` crate,
paired with ``TrackingSettings`` and registered back under its Python
spelling by the binding layer, reaching callers through
:mod:`arco._arco`. Deviation A-19: the crate can bound the metrics
history, and the binding keeps the Python default of retaining every
sample. Deviation A-17: a step validates the elapsed interval instead of
computing with one that is negative, zero, or implausibly large.
"""

from arco._arco import TrackingLoop

__all__ = ["TrackingLoop"]
