"""The Dubins vehicle, re-exported from the compiled extension.

The implementation is ``DubinsVehicle`` in the ``arco-guidance`` crate, registered
back under its Python spelling by the binding layer and reaching callers
through :mod:`arco._arco`. Deviation A-20: the five limit attributes are one `CommandLimits`, and the binding maps both spellings of the two that were renamed.
"""

from arco._arco import DubinsVehicle

__all__ = ["DubinsVehicle"]
