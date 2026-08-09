"""ContinuousPlanner: base class for continuous-space planners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional

import numpy as np

from arco.mapping.occupancy import Occupancy
from arco.planning.continuous.telemetry import (
    PlannerTelemetry,
    write_telemetry,
)
from arco.planning.cost import PlannerCost

TelemetryFn = Callable[[PlannerTelemetry], None]


class ContinuousPlanner(PlannerCost, ABC):
    """Base class for planners operating in continuous state spaces.

    Inherits :meth:`~arco.planning.cost.PlannerCost.distance` and
    :meth:`~arco.planning.cost.PlannerCost.heuristic` from
    :class:`~arco.planning.cost.PlannerCost`.  :meth:`distance` is
    overridden to use step-size-normalized Euclidean distance once a
    subclass sets :attr:`step_size`.

    Pass ``cost=`` to compose an external :class:`PlannerCost` without
    subclassing the algorithm.  When ``None`` (default), the planner uses
    its own distance/heuristic methods.

    Optional ``publisher=``, ``seed=``, and ``rng=`` control telemetry
    sinks and sampling reproducibility.  Defaults preserve historical
    file-telemetry and unseeded RNG behavior.

    Subclasses must implement :meth:`plan`.
    """

    def __init__(
        self,
        occupancy: Occupancy,
        cost: Optional[PlannerCost] = None,
        publisher: Optional[TelemetryFn] = None,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """Initialize the planner with an occupancy map.

        Args:
            occupancy: The occupancy map for collision checking.
            cost: Optional external cost model.  When provided,
                :meth:`distance` and :meth:`heuristic` delegate to it.
            publisher: Optional telemetry sink.  When ``None``, snapshots
                are written with
                :func:`~arco.planning.continuous.telemetry.write_telemetry`.
                Pass
                :func:`~arco.planning.continuous.telemetry.noop_publisher`
                to disable IPC.
            seed: Optional RNG seed used when *rng* is not provided.
            rng: Optional NumPy generator.  When set, takes precedence over
                *seed*.
        """
        self.occupancy = occupancy
        self._cost_model = cost
        self._telemetry_publisher = publisher
        self._seed = seed
        self._rng = rng

    def distance(self, state_a: Any, state_b: Any) -> float:
        """Return step-size-normalized Euclidean distance.

        When an external ``cost`` model was provided at construction,
        delegates to that model instead.

        Args:
            state_a: Origin state.
            state_b: Destination state.

        Returns:
            Non-negative normalized distance.
        """
        if self._cost_model is not None:
            return float(self._cost_model.distance(state_a, state_b))
        step = np.asarray(getattr(self, "step_size", 1.0), dtype=float)
        a = np.asarray(state_a, dtype=float).reshape(-1)
        b = np.asarray(state_b, dtype=float).reshape(-1)
        return float(np.linalg.norm((b - a) / step))

    def heuristic(self, state_a: Any, state_b: Any) -> float:
        """Return remaining-cost estimate, honoring an external cost model.

        Args:
            state_a: Current state.
            state_b: Goal state.

        Returns:
            Heuristic cost estimate.
        """
        if self._cost_model is not None:
            return float(self._cost_model.heuristic(state_a, state_b))
        return self.distance(state_a, state_b)

    def publish_telemetry(self, telemetry: PlannerTelemetry) -> None:
        """Publish a telemetry snapshot via the configured sink.

        Args:
            telemetry: Snapshot to publish.
        """
        if self._telemetry_publisher is not None:
            self._telemetry_publisher(telemetry)
        else:
            write_telemetry(telemetry)

    def make_rng(self) -> np.random.Generator:
        """Return the configured RNG (or a seeded/unseeded default).

        Returns:
            A :class:`numpy.random.Generator` instance.
        """
        if self._rng is not None:
            return self._rng
        return np.random.default_rng(self._seed)

    @abstractmethod
    def plan(
        self, start: np.ndarray, goal: np.ndarray
    ) -> Optional[List[np.ndarray]]:
        """Plan a path from start to goal.

        Args:
            start: The start state as a numpy array.
            goal: The goal state as a numpy array.

        Returns:
            A list of numpy arrays from start to goal, or None if no path
            exists.
        """
