"""Planner-specific scenes for the ARCO unified simulator."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RaceScene(Protocol):
    """Structural protocol for scenes compatible with ``city.run_race()``.

    Any object that satisfies all attributes and methods below can be
    passed to :func:`arco.tools.simulator.main.city.run_race` without
    triggering ``AttributeError`` at runtime.  Mypy / pyright will flag
    missing members at static-analysis time.

    A* attributes (``astar_waypoints``, ``astar_total``, ``astar_metrics``)
    are **not** part of this protocol because they are optional — scenes
    that do not plan with A* simply omit them and ``run_race`` falls back
    to safe defaults via ``getattr``.
    """

    @property
    def title(self) -> str:
        """Window caption for the simulator."""
        ...

    @property
    def bg_color(self) -> tuple[int, int, int]:
        """Background fill colour as an RGB int-triple 0–255."""
        ...

    @property
    def world_points(self) -> list[tuple[float, float]]:
        """Representative world-space points used to compute view bounds."""
        ...

    @property
    def vehicle_config(self) -> Any:
        """Vehicle dynamics / lookahead configuration."""
        ...

    @property
    def rrt_waypoints(self) -> list[tuple[float, float]]:
        """Ordered (x, y) waypoints produced by the RRT* planner."""
        ...

    @property
    def sst_waypoints(self) -> list[tuple[float, float]]:
        """Ordered (x, y) waypoints produced by the SST planner."""
        ...

    @property
    def rrt_total(self) -> int:
        """Number of nodes in the RRT* exploration tree."""
        ...

    @property
    def sst_total(self) -> int:
        """Number of nodes in the SST exploration tree."""
        ...

    @property
    def rrt_metrics(self) -> dict[str, Any]:
        """Planning and trajectory metrics for RRT*."""
        ...

    @property
    def sst_metrics(self) -> dict[str, Any]:
        """Planning and trajectory metrics for SST."""
        ...

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
        ...
