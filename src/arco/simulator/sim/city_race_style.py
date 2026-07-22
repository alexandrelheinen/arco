"""City race glyph and trajectory style constants.

Tuned for the 600 m dark city map so MPC anticipation, rectangular vehicle
bodies, and past traces remain readable in recorded videos.  Racers are
oriented rectangles only — not discs.
"""

from __future__ import annotations

# Vehicle body half-extents in world meters.
# Prior city glyph was 3.0 × 1.4 m; ~2.5× keeps a clear rectangle on the map
# without restoring the old oversized glow blobs.
VEH_HALF_L: float = 8.0
VEH_HALF_W: float = 3.6

# Small Pure-Pursuit carrot disc only (meters).  Never used as the racer glyph.
LOOKAHEAD_DISC_R: float = 1.5

# Past executed trajectory line width in pixels (thicker than the prior 1.5).
PAST_TRACE_WIDTH: float = 3.0

# MPC predicted-horizon polyline width in pixels (thicker than the route).
PREDICTED_TRACE_WIDTH: float = 4.0

# Default city-demo prediction horizon when scenario YAML omits an override.
# 72 × 0.05 s = 3.6 s (~64.8 m at 18 m/s cruise) — about half a city block
# (mean_edge_length = 120 m).  The previous 6.0 s / ~108 m horizon spanned a
# full block and caused A* MPC tracking to cut sharp grid corners and loop.
DEFAULT_CITY_HORIZON_STEP_COUNT: int = 72
DEFAULT_CITY_HORIZON_DT: float = 0.05
