"""Abstract base classes for simulator scenes.

Defines a three-level hierarchy:

* :class:`ArcosimScene` — common root (abstract)
* :class:`SimScene` — single-vehicle scenes (abstract)
* :class:`RaceScene` — multi-vehicle race scenes (abstract)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ArcosimScene(ABC):
    """Common root for all arcosim scene types."""

    @abstractmethod
    def build(self, *, progress=None) -> None:
        """Construct the environment and run the planner.

        Args:
            progress: Optional callable ``(step_name, step_index, total_steps)``.
        """

    @property
    @abstractmethod
    def title(self) -> str:
        """Human-readable scene label."""

    @property
    @abstractmethod
    def bg_color(self) -> tuple[int, int, int]:
        """Background fill color as an RGB tuple."""

    @property
    @abstractmethod
    def world_points(self) -> list[tuple[float, float]]:
        """Representative world-space points for auto-fitting the view."""

    @abstractmethod
    def sidebar_content(
        self, **state: Any
    ) -> list[tuple[list[str], tuple[int, int, int]]]:
        """Return ordered ``(lines, color)`` pairs for the sidebar panel.

        Args:
            **state: Phase-specific keyword arguments. Common keys:
                ``phase``, ``revealed``, ``veh_step``, ``speed``, ``cte``,
                ``finished``, ``paused``.

        Returns:
            Ordered list of ``(lines, color)`` pairs, one per planner section.
        """

    @property
    def background_total(self) -> int:
        """Total background items to reveal; zero skips the reveal phase."""
        return 0

    @property
    def footer_hint(self) -> str:
        """Bottom-bar hint text shown in the chrome overlay."""
        return "SPACE pause  ·  R restart  ·  Q quit"


class SimScene(ArcosimScene):
    """Abstract scene for single-vehicle simulator loop.

    When :attr:`background_total` is zero the loop skips the background-reveal
    phase and starts the vehicle-tracking phase immediately.
    """

    @property
    @abstractmethod
    def zoom_world_points(self) -> list[tuple[float, float]]:
        """World-space points for auto-fitting the zoomed view."""

    @property
    @abstractmethod
    def waypoints(self) -> list[tuple[float, float]]:
        """Ordered ``(x, y)`` waypoints for the vehicle to follow."""

    @property
    @abstractmethod
    def vehicle_config(self) -> Any:
        """Vehicle and controller parameters."""

    @abstractmethod
    def draw_background(self, revealed: int) -> None:
        """Render the static scene background.

        Args:
            revealed: Number of background items revealed so far.
        """


class RaceScene(ArcosimScene):
    """Abstract base class for multi-vehicle race scenes.

    Replaces the old structural Protocol. A* attributes are *not* required
    because they are optional — ``run_race`` uses ``getattr`` fallbacks.
    """

    @property
    @abstractmethod
    def vehicle_config(self) -> Any:
        """Vehicle dynamics / lookahead configuration."""

    @property
    @abstractmethod
    def rrt_waypoints(self) -> list[tuple[float, float]]:
        """Ordered ``(x, y)`` waypoints from the RRT* planner."""

    @property
    @abstractmethod
    def sst_waypoints(self) -> list[tuple[float, float]]:
        """Ordered ``(x, y)`` waypoints from the SST planner."""

    @property
    @abstractmethod
    def rrt_total(self) -> int:
        """Number of nodes in the RRT* exploration tree."""

    @property
    @abstractmethod
    def sst_total(self) -> int:
        """Number of nodes in the SST exploration tree."""

    @property
    @abstractmethod
    def rrt_metrics(self) -> dict[str, Any]:
        """Planning and trajectory metrics for RRT*."""

    @property
    @abstractmethod
    def sst_metrics(self) -> dict[str, Any]:
        """Planning and trajectory metrics for SST."""

    @abstractmethod
    def draw_background(
        self,
        rrt_revealed: int,
        sst_revealed: int,
        racing: bool = False,
    ) -> None:
        """Render the obstacle field and exploration trees.

        Args:
            rrt_revealed: Number of RRT* tree nodes to display.
            sst_revealed: Number of SST tree nodes to display.
            racing: When ``True``, collapse the view to the race backdrop.
        """
