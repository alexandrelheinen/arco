"""City race glyph and trajectory style constants.

Tuned for the 600 m dark city map so MPC anticipation, rectangular vehicle
bodies, and past traces remain readable in recorded videos.  Racers are
oriented rectangles only — not discs.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arco.simulator.sim.tracking import VehicleConfig

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
# 72 × 0.05 s = 3.6 s (~50 m at the soft city cruise of 14 m/s) — about half
# a city block (mean_edge_length = 120 m).
DEFAULT_CITY_HORIZON_STEP_COUNT: int = 72
DEFAULT_CITY_HORIZON_DT: float = 0.05

# Soft city-racer dynamics.  Prior limits (ω=60°/s, ω̇≈∞ at 3600°/s², a=4.9,
# cruise=18) let the NMPC snap onto A* polyline kinks; when that overshoots,
# contouring progress reversed and the car orbited the junction.  These
# softer bounds raise the minimum turn radius above the road half-width so
# sharp corners produce a visible lateral understeer instead of a limit cycle.
CITY_MAX_SPEED: float = 18.0
CITY_CRUISE_SPEED: float = 14.0
CITY_MAX_TURN_RATE_DEG: float = 30.0
CITY_MAX_ACCELERATION: float = 2.5
CITY_MAX_TURN_RATE_DOT_DEG: float = 90.0
CITY_LOOKAHEAD_DISTANCE: float = 28.0
CITY_GOAL_RADIUS: float = 20.0


def make_city_vehicle_config() -> "VehicleConfig":
    """Return soft city-race vehicle / controller limits.

    Returns:
        :class:`~arco.simulator.sim.tracking.VehicleConfig` with turn
        radius larger than the city road half-width at cruise.
    """
    # Local import avoids a cycle with tracking helpers used by scenes.
    from arco.simulator.sim.tracking import VehicleConfig

    return VehicleConfig(
        max_speed=CITY_MAX_SPEED,
        min_speed=0.0,
        cruise_speed=CITY_CRUISE_SPEED,
        lookahead_distance=CITY_LOOKAHEAD_DISTANCE,
        goal_radius=CITY_GOAL_RADIUS,
        max_turn_rate=math.radians(CITY_MAX_TURN_RATE_DEG),
        max_acceleration=CITY_MAX_ACCELERATION,
        max_turn_rate_dot=math.radians(CITY_MAX_TURN_RATE_DOT_DEG),
        curvature_gain=0.0,
    )
