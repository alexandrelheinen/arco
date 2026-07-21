"""City race glyph and trajectory style constants.

Tuned for the 600 m dark city map so MPC anticipation, vehicle bodies, and
past traces remain readable in recorded videos without restoring the old
oversized glow markers.
"""

from __future__ import annotations

# Vehicle body half-extents in world meters (2× the prior 3.0 × 1.4 m glyph).
VEH_HALF_L: float = 6.0
VEH_HALF_W: float = 2.8

# Disc radius for the horizon tip marker (meters).  Sized so the tip reads
# clearly ahead of the vehicle on the 600 m map (~1.5–2 px/m → ~20 px).
LOOKAHEAD_DISC_R: float = 12.0

# Past executed trajectory line width in pixels (thicker than the prior 1.5).
PAST_TRACE_WIDTH: float = 3.0

# MPC predicted-horizon polyline width in pixels (thicker than the route).
PREDICTED_TRACE_WIDTH: float = 4.0

# Default city-demo prediction horizon when scenario YAML omits an override.
# 120 × 0.05 s = 6.0 s (~108 m at 18 m/s cruise) so the horizon tip sits a
# clear road-length ahead of each racer on the recorded video.
DEFAULT_CITY_HORIZON_STEP_COUNT: int = 120
DEFAULT_CITY_HORIZON_DT: float = 0.05
