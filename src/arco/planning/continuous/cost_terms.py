"""Default CostTerm helpers for :class:`TrajectoryOptimizer`.

These related classes share one optimization context contract and together
reproduce the historical five-term composite cost (time, deviation,
velocity, collision+barrier, dynamics).  They are kept in one module
because they are tightly coupled to that shared context and the default
term list factory.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from arco.protocols import CostTerm


class TimeCostTerm:
    """Penalize total traversal time squared: ``w · T²``.

    Args:
        weight: Multiplier for the squared total duration.
    """

    name: str = "time"

    def __init__(self, weight: float) -> None:
        """Initialize the time cost term.

        Args:
            weight: Multiplier for the squared total duration.
        """
        self.weight = float(weight)

    def __call__(self, context: Dict[str, Any]) -> float:
        """Evaluate the time cost.

        Args:
            context: Optimization context; requires ``durs``.

        Returns:
            Weighted squared total duration.
        """
        durs = context["durs"]
        total_time = float(np.sum(durs))
        return self.weight * total_time**2


class DeviationCostTerm:
    """Penalize squared deviation of interior waypoints from the reference.

    Args:
        weight: Multiplier for the summed squared deviation.
    """

    name: str = "deviation"

    def __init__(self, weight: float) -> None:
        """Initialize the deviation cost term.

        Args:
            weight: Multiplier for the summed squared deviation.
        """
        self.weight = float(weight)

    def __call__(self, context: Dict[str, Any]) -> float:
        """Evaluate the path-deviation cost.

        Args:
            context: Optimization context; requires ``pts``, ``ref``, and
                ``segment_count``.

        Returns:
            Weighted sum of squared interior deviations, or ``0.0`` when
            there are no interior waypoints.
        """
        segment_count = int(context["segment_count"])
        interior_count = segment_count - 1
        if interior_count <= 0:
            return 0.0
        pts = context["pts"]
        ref = context["ref"]
        ref_interior = np.array(ref[1:-1])
        pts_interior = pts[1:-1]
        return self.weight * float(np.sum((pts_interior - ref_interior) ** 2))


class VelocityCostTerm:
    """Penalize squared deviation of segment speeds from cruise speed.

    Args:
        weight: Multiplier for the summed squared speed error.
        cruise_speed: Target traversal speed (world units / s).
    """

    name: str = "velocity"

    def __init__(self, weight: float, cruise_speed: float) -> None:
        """Initialize the velocity cost term.

        Args:
            weight: Multiplier for the summed squared speed error.
            cruise_speed: Target traversal speed (world units / s).
        """
        self.weight = float(weight)
        self.cruise_speed = float(cruise_speed)

    def __call__(self, context: Dict[str, Any]) -> float:
        """Evaluate the velocity-tracking cost.

        Args:
            context: Optimization context; requires ``speeds``.

        Returns:
            Weighted sum of squared ``(speed − cruise)`` errors.
        """
        speeds = context["speeds"]
        return self.weight * float(np.sum((speeds - self.cruise_speed) ** 2))


class CollisionCostTerm:
    """Soft clearance penalty plus barrier-style penetration growth.

    Combines the quadratic clearance violation with the scaled barrier
    term used historically inside ``TrajectoryOptimizer._cost``.

    Args:
        weight: Multiplier applied to both soft and barrier penalties.
        barrier_scale: Extra multiplier for the barrier contribution.
        barrier_power: Exponent on normalized penetration depth.
    """

    name: str = "collision"

    def __init__(
        self,
        weight: float,
        barrier_scale: float = 50.0,
        barrier_power: float = 4.0,
    ) -> None:
        """Initialize the collision cost term.

        Args:
            weight: Multiplier applied to both soft and barrier penalties.
            barrier_scale: Extra multiplier for the barrier contribution.
            barrier_power: Exponent on normalized penetration depth.
        """
        self.weight = float(weight)
        self.barrier_scale = float(barrier_scale)
        self.barrier_power = float(barrier_power)

    def __call__(self, context: Dict[str, Any]) -> float:
        """Evaluate soft collision plus barrier penalties.

        Args:
            context: Optimization context; requires ``pts``,
                ``segment_count``, ``occupancy``, and ``sample_count``.

        Returns:
            Sum of the soft clearance penalty and the barrier penalty.
        """
        pts = context["pts"]
        segment_count = int(context["segment_count"])
        occupancy = context["occupancy"]
        sample_count = int(context["sample_count"])
        interior_count = segment_count - 1

        clearance = getattr(occupancy, "clearance", 0.5)
        j_collision = 0.0
        j_collision_barrier = 0.0

        query_pts_list: list[np.ndarray] = []
        if interior_count > 0:
            query_pts_list.append(pts[1:-1])

        if sample_count > 0:
            for i in range(segment_count):
                p_a = pts[i]
                p_b = pts[i + 1]
                alphas = np.linspace(0.0, 1.0, sample_count + 2)[1:-1]
                samples = p_a + alphas[:, None] * (p_b - p_a)
                query_pts_list.append(samples)

        if query_pts_list:
            all_query = np.concatenate(query_pts_list, axis=0)
            if hasattr(occupancy, "query_distances"):
                dists = occupancy.query_distances(all_query)
            else:
                dists = np.array(
                    [occupancy.nearest_obstacle(p)[0] for p in all_query]
                )
            penetrations = np.maximum(0.0, clearance - dists)
            j_collision = self.weight * float(np.sum(penetrations**2))
            clearance_safe = max(float(clearance), 1e-9)
            normalized_penetrations = penetrations / clearance_safe
            j_collision_barrier = (
                self.weight
                * self.barrier_scale
                * float(np.sum(normalized_penetrations**self.barrier_power))
            )

        return j_collision + j_collision_barrier


class DynamicsCostTerm:
    """Penalize implied segment speeds outside ``[min_speed, max_speed]``.

    Args:
        weight: Multiplier for the summed squared bound violations.
        max_speed: Optional upper speed limit (world units / s).
        min_speed: Optional lower speed limit (world units / s).
    """

    name: str = "dynamics"

    def __init__(
        self,
        weight: float,
        max_speed: Optional[float] = None,
        min_speed: Optional[float] = None,
    ) -> None:
        """Initialize the dynamics cost term.

        Args:
            weight: Multiplier for the summed squared bound violations.
            max_speed: Optional upper speed limit (world units / s).
            min_speed: Optional lower speed limit (world units / s).
        """
        self.weight = float(weight)
        self.max_speed = max_speed
        self.min_speed = min_speed

    def __call__(self, context: Dict[str, Any]) -> float:
        """Evaluate the dynamics-bound penalty.

        Args:
            context: Optimization context; requires ``speeds``.

        Returns:
            Weighted sum of squared speed-bound violations, or ``0.0``
            when both bounds are unset.
        """
        if self.max_speed is None and self.min_speed is None:
            return 0.0
        speeds = context["speeds"]
        j_dynamics = 0.0
        if self.max_speed is not None:
            over = np.maximum(0.0, speeds - self.max_speed)
            j_dynamics += float(np.sum(over**2))
        if self.min_speed is not None:
            under = np.maximum(0.0, self.min_speed - speeds)
            j_dynamics += float(np.sum(under**2))
        return j_dynamics * self.weight


def build_default_cost_terms(
    *,
    weight_time: float,
    weight_deviation: float,
    weight_velocity: float,
    weight_collision: float,
    weight_dynamics: float,
    cruise_speed: float,
    collision_barrier_scale: float,
    collision_barrier_power: float,
    max_speed: Optional[float],
    min_speed: Optional[float],
) -> List[CostTerm]:
    """Build the five historical default optimizer cost terms.

    Args:
        weight_time: Weight for the total-time-squared term.
        weight_deviation: Weight for interior path deviation.
        weight_velocity: Weight for cruise-speed tracking.
        weight_collision: Weight for clearance / barrier penalties.
        weight_dynamics: Weight for speed-bound penalties.
        cruise_speed: Target traversal speed (world units / s).
        collision_barrier_scale: Barrier multiplier on clearance violations.
        collision_barrier_power: Barrier exponent on normalized penetration.
        max_speed: Optional upper speed limit for the dynamics term.
        min_speed: Optional lower speed limit for the dynamics term.

    Returns:
        Ordered list of five :class:`~arco.protocols.CostTerm` instances
        matching historical ``_cost`` composition order.
    """
    return [
        TimeCostTerm(weight_time),
        DeviationCostTerm(weight_deviation),
        VelocityCostTerm(weight_velocity, cruise_speed),
        CollisionCostTerm(
            weight_collision,
            barrier_scale=collision_barrier_scale,
            barrier_power=collision_barrier_power,
        ),
        DynamicsCostTerm(
            weight_dynamics,
            max_speed=max_speed,
            min_speed=min_speed,
        ),
    ]
