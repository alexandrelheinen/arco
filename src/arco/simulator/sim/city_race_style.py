"""City race glyph and trajectory style constants.

Tuned for the 600 m dark city map so MPC anticipation, vehicle bodies, and
past traces remain readable in recorded videos without restoring the old
oversized glow markers.
"""

from __future__ import annotations

# Vehicle body half-extents in world meters (2× the prior 3.0 × 1.4 m glyph).
VEH_HALF_L: float = 6.0
VEH_HALF_W: float = 2.8

# Disc radius for the horizon tip / lookahead marker (meters).
LOOKAHEAD_DISC_R: float = 2.0

# Past executed trajectory line width in pixels (slightly thicker than 1.5).
PAST_TRACE_WIDTH: float = 2.5

# MPC predicted-horizon polyline width in pixels.
PREDICTED_TRACE_WIDTH: float = 2.0

# Default city-demo prediction horizon when scenario YAML omits an override.
# 60 × 0.05 s = 3.0 s (~54 m at 18 m/s cruise) so anticipation reads on video.
DEFAULT_CITY_HORIZON_STEP_COUNT: int = 60
DEFAULT_CITY_HORIZON_DT: float = 0.05
