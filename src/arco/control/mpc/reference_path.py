"""ReferencePath, re-exported from the compiled extension.

The implementation is ``ReferencePath`` in the ``arco-control`` crate.
Zero-length segments are dropped on the way in and every query clamps its
arc length into the path, which is what lets a controller ask about a
horizon running off the end of the route.
"""

from arco._arco import ReferencePath

__all__ = ["ReferencePath"]
