"""Moving-average smoothing, re-exported from the compiled extension.

The implementation is ``MovingAverageInterpolator`` in the ``arco-guidance`` crate, registered
back under its Python spelling by the binding layer and reaching callers
through :mod:`arco._arco`.
"""

from arco._arco import MovingAverageInterpolator

__all__ = ["MovingAverageInterpolator"]
