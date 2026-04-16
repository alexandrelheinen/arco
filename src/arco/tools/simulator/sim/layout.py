"""Two-column screen layout geometry for arcosim.

Defines :class:`ScreenLayout` (geometry) and helpers for drawing the
header / footer / sidebar chrome and the sidebar text panel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pygame


@dataclass
class ScreenLayout:
    """Immutable two-column screen layout for the arcosim simulator.

    Divides the display into a left sidebar, a content area, and thin header /
    footer bars:

    .. code-block:: text

        ┌───────────────────────────────────────┐  ← header_h (40 px)
        │               Header                  │
        ├─────────────┬─────────────────────────┤
        │             │                         │
        │   Sidebar   │       Content           │
        │  (sidebar_w)│  (content_w × content_h)│
        │             │                         │
        ├─────────────┴─────────────────────────┤  ← footer_h (30 px)
        │               Footer                  │
        └───────────────────────────────────────┘

    Args:
        sw: Screen width in pixels.
        sh: Screen height in pixels.
        sidebar_w: Width of the left legend / menu column in pixels.
        header_h: Height of the top title bar in pixels.
        footer_h: Height of the bottom hint bar in pixels.
    """

    sw: int
    sh: int
    sidebar_w: int = 260
    header_h: int = 40
    footer_h: int = 30

    @property
    def content_x(self) -> int:
        """Left edge of the content area in screen pixels."""
        return self.sidebar_w

    @property
    def content_y(self) -> int:
        """Bottom edge of the content area in OpenGL pixel coordinates.

        OpenGL places ``y=0`` at the bottom of the window, so the content
        area starts just above the footer bar.
        """
        return self.footer_h

    @property
    def content_w(self) -> int:
        """Width of the content area in pixels (minimum 1)."""
        return max(1, self.sw - self.sidebar_w)

    @property
    def content_h(self) -> int:
        """Height of the content area in pixels (minimum 1)."""
        return max(1, self.sh - self.header_h - self.footer_h)

    def setup_content_viewport(self) -> None:
        """Restrict the OpenGL viewport to the content area.

        Must be called before any world-space GL draw calls so they are
        clipped to the right-hand content column.
        """
        from OpenGL.GL import glViewport  # type: ignore[import-untyped]

        glViewport(
            self.content_x, self.content_y, self.content_w, self.content_h
        )

    def reset_viewport(self) -> None:
        """Restore the OpenGL viewport to the full window.

        Must be called after world-space GL draws and before 2-D overlay
        rendering so overlays can paint anywhere on screen.
        """
        from OpenGL.GL import glViewport  # type: ignore[import-untyped]

        glViewport(0, 0, self.sw, self.sh)


def make_chrome_surface(
    layout: ScreenLayout,
    title: str,
    footer_hint: str,
    title_font: Any,
    hint_font: Any,
) -> Any:
    """Build a full-screen translucent chrome overlay surface.

    Draws the header bar (top), footer bar (bottom), and sidebar background
    (left).  The content area is left fully transparent.

    Args:
        layout: Screen geometry descriptor.
        title: Scene title rendered centered in the header bar.
        footer_hint: Short hint text rendered centered in the footer bar.
        title_font: Pygame font used to render *title*.
        hint_font: Pygame font used to render *footer_hint*.

    Returns:
        A ``pygame.Surface`` with ``SRCALPHA`` pixel format, sized
        ``layout.sw × layout.sh``.
    """
    import pygame

    sw, sh = layout.sw, layout.sh
    sidebar_w = layout.sidebar_w
    header_h = layout.header_h
    footer_h = layout.footer_h

    _C_DARK: tuple[int, int, int, int] = (14, 16, 26, 235)
    _C_SIDEBAR: tuple[int, int, int, int] = (10, 12, 20, 235)
    _C_BORDER: tuple[int, int, int, int] = (55, 70, 100, 255)
    _C_TITLE: tuple[int, int, int] = (200, 210, 230)
    _C_HINT: tuple[int, int, int] = (100, 115, 140)

    surf = pygame.Surface((sw, sh), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))

    # Header bar
    pygame.draw.rect(surf, _C_DARK, pygame.Rect(0, 0, sw, header_h))
    pygame.draw.line(surf, _C_BORDER, (0, header_h - 1), (sw, header_h - 1))
    if title:
        title_surf = title_font.render(title, True, _C_TITLE)
        tx = (sw - title_surf.get_width()) // 2
        ty = (header_h - title_surf.get_height()) // 2
        surf.blit(title_surf, (tx, ty))

    # Footer bar
    fy = sh - footer_h
    pygame.draw.rect(surf, _C_DARK, pygame.Rect(0, fy, sw, footer_h))
    pygame.draw.line(surf, _C_BORDER, (0, fy), (sw, fy))
    if footer_hint:
        hint_surf = hint_font.render(footer_hint, True, _C_HINT)
        hx = (sw - hint_surf.get_width()) // 2
        hy = fy + (footer_h - hint_surf.get_height()) // 2
        surf.blit(hint_surf, (hx, hy))

    # Sidebar background (between header and footer)
    pygame.draw.rect(
        surf,
        _C_SIDEBAR,
        pygame.Rect(0, header_h, sidebar_w, sh - header_h - footer_h),
    )
    pygame.draw.line(
        surf,
        _C_BORDER,
        (sidebar_w - 1, header_h),
        (sidebar_w - 1, sh - footer_h),
    )

    return surf


def draw_sidebar_panel(
    layout: ScreenLayout,
    font: Any,
    sections: list[tuple[list[str], tuple[int, int, int]]],
    sw: int,
    sh: int,
) -> None:
    """Render planner-info sections as colored text in the sidebar.

    Each section is a ``(lines, color)`` pair. Sections are stacked
    vertically with a blank spacer row between them. Text is rendered with
    a subtle drop-shadow for readability against the dark sidebar.

    Args:
        layout: Screen geometry descriptor.
        font: Pygame monospace font.
        sections: Ordered list of ``(lines, color)`` pairs.
        sw: Screen width in pixels.
        sh: Screen height in pixels.
    """
    if not sections:
        return

    import pygame

    from arco.config.palette import ui_rgb
    from arco.tools.simulator import renderer_gl

    _C_SHADOW = ui_rgb("hud_shadow")

    padding = 8
    panel_w = layout.sidebar_w - 2 * padding
    lh = font.get_linesize() + 2

    entries: list[tuple[str, tuple[int, int, int] | None]] = []
    for i, (lines, color) in enumerate(sections):
        for line in lines:
            entries.append((line, color))
        if i < len(sections) - 1:
            entries.append(("", None))

    panel_h = len(entries) * lh + padding * 2
    surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    y = padding
    for text, color in entries:
        if not text or color is None:
            y += lh
            continue
        surf.blit(font.render(text, True, _C_SHADOW), (1, y + 1))
        surf.blit(font.render(text, True, color), (0, y))
        y += lh

    renderer_gl.blit_overlay(surf, padding, layout.header_h + padding, sw, sh)
