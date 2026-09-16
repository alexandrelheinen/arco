"""ARCO illustration gallery: 16:9 plates rendered from real solver runs.

Every image this package produces is drawn from output of the shipped
``arco`` planners and controllers — no traced curves, no mock data.  The
entry point is ``tools/render_gallery.py``.
"""

from __future__ import annotations

__all__ = ["canvas", "plates", "solution", "theme", "world"]
