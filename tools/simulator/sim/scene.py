"""Abstract base class for simulator scenes.

Every planner-specific scene implements :class:`SimScene` to provide the
unified interface consumed by :func:`~sim.loop.run_sim`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pygame
from sim.tracking import VehicleConfig


class SimScene(ABC):
    """Abstract scene consumed by the unified simulator loop.

    A scene encapsulates all planner-specific logic:

    * environment construction and path planning (in :meth:`build`)
    * static background rendering (:meth:`draw_background`)
    * planning-phase HUD rendering (:meth:`draw_background_hud`)

    The loop calls :meth:`build` **after** ``pygame.init()``, so scenes may
    safely call ``pygame.font.SysFont`` inside :meth:`build`.

    When :attr:`background_total` is zero the loop skips the background-reveal
    phase and starts the vehicle-tracking phase immediately.
    """

    @abstractmethod
    def build(self) -> None:
        """Construct the environment and run the planner.

        Called once, after ``pygame.init()``.
        """

    @property
    @abstractmethod
    def title(self) -> str:
        """Human-readable scene label shown in the HUD."""

    @property
    @abstractmethod
    def bg_color(self) -> tuple[int, int, int]:
        """Background fill colour as an RGB tuple."""

    @property
    @abstractmethod
    def world_points(self) -> list[tuple[float, float]]:
        """Representative world-space points used to auto-fit the full view."""

    @property
    @abstractmethod
    def zoom_world_points(self) -> list[tuple[float, float]]:
        """Representative world-space points used to auto-fit the zoomed view."""

    @property
    @abstractmethod
    def waypoints(self) -> list[tuple[float, float]]:
        """Ordered ``(x, y)`` waypoints for the vehicle to follow."""

    @property
    @abstractmethod
    def vehicle_config(self) -> VehicleConfig:
        """Vehicle and controller parameters."""

    @property
    @abstractmethod
    def background_total(self) -> int:
        """Total number of background items to reveal.

        Zero means skip the background-reveal phase entirely.
        """

    @abstractmethod
    def draw_background(
        self,
        surface: pygame.Surface,
        transform: object,
        revealed: int,
    ) -> None:
        """Render the static scene background.

        Called every frame during both the background-reveal and
        vehicle-tracking phases.

        Args:
            surface: Pygame surface to draw onto.
            transform: Callable ``(wx, wy) -> (sx, sy)`` from
                :class:`~sim.camera.FollowTransform` or equivalent.
            revealed: Number of background items revealed so far.
        """

    @abstractmethod
    def draw_background_hud(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        revealed: int,
    ) -> None:
        """Render the background-phase HUD overlay.

        Called only during the background-reveal phase.

        Args:
            surface: Pygame surface to draw onto.
            font: Monospace font for HUD text.
            revealed: Number of background items revealed so far.
        """
