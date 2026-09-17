"""Joint-space tracking, re-exported from the compiled extension.

The implementation is ``JointSpaceTracker`` in the ``arco-control`` crate,
paired with ``JointTrackerSettings`` and registered back under its Python
spelling by the binding layer, reaching callers through
:mod:`arco._arco`. The tracker drives joint positions toward a target
under velocity and acceleration limits, with an optional repulsion term
read from an occupancy map. Deviation A-17: a step validates the elapsed
interval.
"""

from arco._arco import JointSpaceTracker

__all__ = ["JointSpaceTracker"]
