"""B-spline interpolation, re-exported from the compiled extension.

The implementation is ``BSplineInterpolator`` in the ``arco-guidance`` crate, registered
back under its Python spelling by the binding layer and reaching callers
through :mod:`arco._arco`. Deviation C-12: this returns its input unchanged, in Python as well, and is documented as a placeholder rather than an implementation.
"""

from arco._arco import BSplineInterpolator

__all__ = ["BSplineInterpolator"]
