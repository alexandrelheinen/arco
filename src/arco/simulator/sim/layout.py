"""Shared arcosim presentation chrome (header / sidebar / footer).

All four primary scenarios (city, ppp, rrp, occ) compose their on-screen
shell through :class:`ScreenLayout` and the helpers below so release videos
share one layout language: left legend, phase title, method accent stripe,
footer hints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    import pygame

# Default chrome geometry — taller header + wider sidebar than the old 40/260.
_DEFAULT_SIDEBAR_W = 280
_DEFAULT_HEADER_H = 44
_DEFAULT_FOOTER_H = 32
_METHOD_STRIPE_H = 3


@dataclass
class ScreenLayout:
    """Immutable two-column screen layout for the arcosim simulator.

    Divides the display into a left sidebar, a content area, and thin header /
    footer bars:

    .. code-block:: text

        +---------------------------------------+  <- header (+ method stripe)
        |  Title                                |
        +-------------+-------------------------+
        |             |                         |
        |   Sidebar   |       Content           |
        |             |                         |
        +-------------+-------------------------+  <- footer
        |               Footer                  |
        +---------------------------------------+

    Args:
        sw: Screen width in pixels.
        sh: Screen height in pixels.
        sidebar_w: Width of the left legend / menu column in pixels.
        header_h: Height of the top title bar in pixels.
        footer_h: Height of the bottom hint bar in pixels.
    """

    sw: int
    sh: int
    sidebar_w: int = _DEFAULT_SIDEBAR_W
    header_h: int = _DEFAULT_HEADER_H
    footer_h: int = _DEFAULT_FOOTER_H

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


def scenario_phase_title(scenario: str, phase: str) -> str:
    """Return a short header title for a scenario presentation phase.

    Args:
        scenario: One of ``city``, ``ppp``, ``rrp``, ``occ``.
        phase: Scenario-local phase key (``background`` / ``racing`` /
            ``show`` / ``race`` / ``done`` / ``running`` …).

    Returns:
        Title string for the chrome header.
    """
    key = (scenario.lower().strip(), phase.lower().strip())
    table: dict[tuple[str, str], str] = {
        ("city", "background"): "City  ·  planning reveal",
        ("city", "racing"): "City  ·  race",
        ("city", "done"): "City  ·  finished",
        ("ppp", "show"): "PPP  ·  path reveal",
        ("ppp", "race"): "PPP  ·  race",
        ("ppp", "done"): "PPP  ·  finished",
        ("rrp", "show"): "RRP  ·  path reveal",
        ("rrp", "race"): "RRP  ·  race",
        ("rrp", "done"): "RRP  ·  finished",
        ("occ", "running"): "OCC  ·  piano movers  ·  RRT* ‖ SST",
        ("occ", "paused"): "OCC  ·  paused  ·  RRT* ‖ SST",
    }
    if key in table:
        return table[key]
    return f"{scenario.upper()}  ·  {phase}"


def build_compact_planner_sections(
    planners: Sequence[tuple[str, tuple[int, int, int], dict[str, Any]]],
) -> list[tuple[list[str], tuple[int, int, int]]]:
    """Build short sidebar sections for planning / path-reveal phases.

    Args:
        planners: ``(name, color, metrics)`` rows.  Metrics may include
            ``nodes``, ``steps``, ``planner_time``, ``planned_path_length``,
            ``path_status``.

    Returns:
        Sidebar ``(lines, color)`` sections.
    """

    def _clock(seconds: float) -> str:
        rounded = int(round(float(seconds)))
        mins, secs = divmod(rounded, 60)
        return f"{mins:02d}min{secs:02d}s"

    sections: list[tuple[list[str], tuple[int, int, int]]] = []
    for name, color, metrics in planners:
        nodes = int(metrics.get("nodes", 0))
        steps = int(metrics.get("steps", 0))
        path_m = int(round(float(metrics.get("planned_path_length", 0.0))))
        plan_t = _clock(float(metrics.get("planner_time", 0.0)))
        status = str(metrics.get("path_status", "n/a"))
        sections.append(
            (
                [
                    name,
                    f"  {nodes} nodes · {steps} steps",
                    f"  plan {plan_t}",
                    f"  path {path_m} m · {status}",
                ],
                color,
            )
        )
    return sections


def make_chrome_surface(
    layout: ScreenLayout,
    title: str,
    footer_hint: str,
    title_font: Any,
    hint_font: Any,
    *,
    method_colors: Sequence[tuple[int, int, int]] = (),
) -> Any:
    """Build a full-screen translucent chrome overlay surface.

    Draws the header bar (left-aligned title + optional method accent
    stripe), footer bar, and sidebar background.  The content area is left
    fully transparent.  Colors come from ``ui.chrome_*`` in ``colors.yml``.

    Args:
        layout: Screen geometry descriptor.
        title: Scene title rendered in the header bar.
        footer_hint: Short hint text rendered centered in the footer bar.
        title_font: Pygame font used to render *title*.
        hint_font: Pygame font used to render *footer_hint*.
        method_colors: Optional method RGB colors drawn as a thin stripe
            under the header (one segment each).

    Returns:
        A ``pygame.Surface`` with ``SRCALPHA`` pixel format, sized
        ``layout.sw × layout.sh``.
    """
    import pygame

    from arco.config.palette import ui_rgb

    sw, sh = layout.sw, layout.sh
    sidebar_w = layout.sidebar_w
    header_h = layout.header_h
    footer_h = layout.footer_h

    bar = ui_rgb("chrome_bar")
    side = ui_rgb("chrome_sidebar")
    border = ui_rgb("chrome_border")
    title_c = ui_rgb("chrome_title")
    hint_c = ui_rgb("chrome_hint")

    _C_DARK = (bar[0], bar[1], bar[2], 242)
    _C_SIDEBAR = (side[0], side[1], side[2], 242)
    _C_BORDER = (border[0], border[1], border[2], 255)

    surf = pygame.Surface((sw, sh), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))

    # Header bar
    pygame.draw.rect(surf, _C_DARK, pygame.Rect(0, 0, sw, header_h))
    if title:
        title_surf = title_font.render(title, True, title_c)
        tx = 16
        ty = (header_h - _METHOD_STRIPE_H - title_surf.get_height()) // 2
        surf.blit(title_surf, (tx, max(2, ty)))

    # Method accent stripe under the header (reads as a race identity bar).
    stripe_y = header_h - _METHOD_STRIPE_H
    if method_colors:
        n = len(method_colors)
        seg_w = max(1, sw // n)
        for i, color in enumerate(method_colors):
            x0 = i * seg_w
            w = sw - x0 if i == n - 1 else seg_w
            pygame.draw.rect(
                surf,
                (color[0], color[1], color[2], 255),
                pygame.Rect(x0, stripe_y, w, _METHOD_STRIPE_H),
            )
    else:
        pygame.draw.line(
            surf, _C_BORDER, (0, header_h - 1), (sw, header_h - 1)
        )

    # Footer bar
    fy = sh - footer_h
    pygame.draw.rect(surf, _C_DARK, pygame.Rect(0, fy, sw, footer_h))
    pygame.draw.line(surf, _C_BORDER, (0, fy), (sw, fy))
    if footer_hint:
        hint_surf = hint_font.render(footer_hint, True, hint_c)
        hx = (sw - hint_surf.get_width()) // 2
        hy = fy + (footer_h - hint_surf.get_height()) // 2
        surf.blit(hint_surf, (hx, hy))

    # Sidebar background (between header and footer) — skip when width is 0.
    if sidebar_w > 0:
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


def paint_sidebar_panel(
    target: Any,
    layout: ScreenLayout,
    font: Any,
    sections: list[tuple[list[str], tuple[int, int, int]]],
) -> None:
    """Paint planner-info sections onto a pygame *target* surface.

    Used by pure-pygame scenarios (OCC).  OpenGL scenarios should call
    :func:`draw_sidebar_panel` instead.

    Args:
        target: Destination ``pygame.Surface`` (usually the display).
        layout: Screen geometry descriptor.
        font: Pygame monospace font.
        sections: Ordered list of ``(lines, color)`` pairs.
    """
    if not sections or layout.sidebar_w <= 0:
        return

    import pygame

    from arco.config.palette import ui_rgb

    _C_SHADOW = ui_rgb("hud_shadow")
    padding = 10
    accent_w = 3
    panel_w = layout.sidebar_w - 2 * padding
    lh = font.get_linesize() + 2

    y = layout.header_h + padding
    for i, (lines, color) in enumerate(sections):
        block_h = len(lines) * lh + 6
        pygame.draw.rect(
            target,
            color,
            pygame.Rect(padding, y, accent_w, max(block_h - 4, 8)),
        )
        text_x = padding + accent_w + 6
        ty = y
        for line in lines:
            target.blit(
                font.render(line, True, _C_SHADOW), (text_x + 1, ty + 1)
            )
            target.blit(font.render(line, True, color), (text_x, ty))
            ty += lh
        y = ty + (14 if i < len(sections) - 1 else 0)


def draw_sidebar_panel(
    layout: ScreenLayout,
    font: Any,
    sections: list[tuple[list[str], tuple[int, int, int]]],
    sw: int,
    sh: int,
) -> None:
    """Render planner-info sections as colored text in the sidebar (OpenGL).

    Each section is a ``(lines, color)`` pair with a 3 px method accent bar
    on the left.  Text uses a subtle drop-shadow for readability.

    Args:
        layout: Screen geometry descriptor.
        font: Pygame monospace font.
        sections: Ordered list of ``(lines, color)`` pairs.
        sw: Screen width in pixels.
        sh: Screen height in pixels.
    """
    if not sections or layout.sidebar_w <= 0:
        return

    import pygame

    from arco.config.palette import ui_rgb
    from arco.simulator import renderer_gl

    _C_SHADOW = ui_rgb("hud_shadow")

    padding = 10
    accent_w = 3
    panel_w = layout.sidebar_w - 2 * padding
    lh = font.get_linesize() + 2

    # Measure total height.
    total_h = padding
    for i, (lines, _color) in enumerate(sections):
        total_h += len(lines) * lh + 6
        if i < len(sections) - 1:
            total_h += 14
    total_h += padding

    surf = pygame.Surface((panel_w, total_h), pygame.SRCALPHA)
    y = padding
    for i, (lines, color) in enumerate(sections):
        block_h = len(lines) * lh + 6
        pygame.draw.rect(
            surf,
            color,
            pygame.Rect(0, y, accent_w, max(block_h - 4, 8)),
        )
        text_x = accent_w + 6
        ty = y
        for line in lines:
            surf.blit(font.render(line, True, _C_SHADOW), (text_x + 1, ty + 1))
            surf.blit(font.render(line, True, color), (text_x, ty))
            ty += lh
        y = ty + (14 if i < len(sections) - 1 else 0)

    renderer_gl.blit_overlay(surf, padding, layout.header_h + padding, sw, sh)
