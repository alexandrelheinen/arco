"""Structural protocols for ARCO extension points.

These typing protocols document duck-typed contracts already used across
mapping, planning, guidance, and control.  They impose no runtime
behavior change; existing classes satisfy them structurally.
"""

from .avoidance import AvoidanceStrategy
from .cost_term import CostTerm
from .discrete_map import DiscreteMap
from .occupancy import OccupancyLike
from .optimizer import OptimizerLike
from .path_tracker import PathTracker
from .planner import PlannerLike
from .pruner import PrunerLike
from .sampler import Sampler
from .segment_checker import SegmentChecker
from .steerer import Steerer
from .telemetry import TelemetryPublisher
from .vehicle import VehicleModel

__all__ = [
    "AvoidanceStrategy",
    "CostTerm",
    "DiscreteMap",
    "OccupancyLike",
    "OptimizerLike",
    "PathTracker",
    "PlannerLike",
    "PrunerLike",
    "Sampler",
    "SegmentChecker",
    "Steerer",
    "TelemetryPublisher",
    "VehicleModel",
]
