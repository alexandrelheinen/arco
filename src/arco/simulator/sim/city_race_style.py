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

# Past executed trajectory line width in pixels — softer than the route so
# the planned lane stays readable under three overlapping racers.
PAST_TRACE_WIDTH: float = 2.5

# MPC predicted-horizon polyline width in pixels (slightly thicker than past).
PREDICTED_TRACE_WIDTH: float = 3.5

# Default city-demo prediction horizon when scenario YAML omits an override.
# 72 × 0.05 s = 3.6 s (~43 m at the soft city cruise of 12 m/s) — about half
# a city block (mean_edge_length = 120 m).
DEFAULT_CITY_HORIZON_STEP_COUNT: int = 72
DEFAULT_CITY_HORIZON_DT: float = 0.05

# Soft but lane-viable city-racer dynamics.  Prior snap limits (ω=60°/s,
# ω̇≈∞, a=4.9, cruise=18) let the NMPC nail A* kinks then reverse progress.
# The first soft pass (ω=30°/s, cruise=14 → R_min≈27 m) made understeer
# visible but *forced* corner cuts outside the 15 m road half-width when
# curve-speed limiting failed.  These bounds keep turns soft while a
# corrected polyline κ + small contour deadzone let the car slow and widen
# *inside* the navigable lane.
CITY_MAX_SPEED: float = 16.0
CITY_CRUISE_SPEED: float = 12.0
CITY_MAX_TURN_RATE_DEG: float = 40.0
CITY_MAX_ACCELERATION: float = 2.5
CITY_MAX_TURN_RATE_DOT_DEG: float = 90.0
CITY_LOOKAHEAD_DISTANCE: float = 28.0
CITY_GOAL_RADIUS: float = 20.0
# City road half-width (m); used by tests / docs as the lane budget.
CITY_ROAD_HALF_WIDTH: float = 15.0


def make_city_vehicle_config() -> "VehicleConfig":
    """Return soft, lane-viable city-race vehicle / controller limits.

    Returns:
        :class:`~arco.simulator.sim.tracking.VehicleConfig` with turn
        rate and cruise chosen so curve-limited corner radii fit inside
        the city road half-width.
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
