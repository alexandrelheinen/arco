"""City race presentation: follow-cam framing, standings HUD, minimap.

Planning reveal stays on the full neighborhood map.  During the race the
main viewport zooms to the pack (smooth chase) and a corner minimap keeps
the full city readable — a real presentation change vs a static bird's-eye.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    import pygame

# Zoomed race window half-size floor (m).  ~200 m across ≈ 1.5 city blocks.
RACE_VIEW_MIN_HALF_EXTENT: float = 100.0
# Extra pad around the pack bounding box before choosing half-extent (m).
RACE_VIEW_MARGIN: float = 45.0
# Critically-damped chase — slower than the default 3 rad/s for a calmer race feel.
RACE_CAMERA_NATURAL_FREQUENCY: float = 1.8
# Corner minimap (pixels) and inset from the content edges.
MINIMAP_SIZE_PX: int = 176
MINIMAP_PAD_PX: int = 12


def pack_centroid(
    positions: Sequence[tuple[float, float]],
) -> tuple[float, float]:
    """Return the mean XY of *positions*, or ``(0, 0)`` if empty.

    Args:
        positions: World-frame ``(x, y)`` samples (active racers).

    Returns:
        Centroid ``(x, y)`` in meters.
    """
    if not positions:
        return (0.0, 0.0)
    n = float(len(positions))
    return (
        sum(p[0] for p in positions) / n,
        sum(p[1] for p in positions) / n,
    )


def race_view_bounds(
    center_x: float,
    center_y: float,
    positions: Sequence[tuple[float, float]],
    *,
    world_x_min: float,
    world_x_max: float,
    world_y_min: float,
    world_y_max: float,
    min_half_extent: float = RACE_VIEW_MIN_HALF_EXTENT,
    margin: float = RACE_VIEW_MARGIN,
) -> tuple[float, float, float, float]:
    """Compute a square orthographic window around the chase camera.

    The half-extent is at least *min_half_extent* and grows to fit the pack
    plus *margin*.  The window is then shifted to stay inside the world box
    when possible.

    Args:
        center_x: Filtered camera center x (m).
        center_y: Filtered camera center y (m).
        positions: Racer positions used to size the window.
        world_x_min: World left edge (m).
        world_x_max: World right edge (m).
        world_y_min: World bottom edge (m).
        world_y_max: World top edge (m).
        min_half_extent: Minimum half-width / half-height (m).
        margin: Extra pad around the pack AABB (m).

    Returns:
        ``(x_min, x_max, y_min, y_max)`` in world meters.
    """
    half = float(min_half_extent)
    if positions:
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        span = max(max(xs) - min(xs), max(ys) - min(ys))
        half = max(half, 0.5 * span + float(margin))

    world_w = world_x_max - world_x_min
    world_h = world_y_max - world_y_min
    half = min(half, 0.5 * max(world_w, world_h))

    x0 = center_x - half
    x1 = center_x + half
    y0 = center_y - half
    y1 = center_y + half

    if x0 < world_x_min:
        shift = world_x_min - x0
        x0 += shift
        x1 += shift
    if x1 > world_x_max:
        shift = x1 - world_x_max
        x0 -= shift
        x1 -= shift
    if y0 < world_y_min:
        shift = world_y_min - y0
        y0 += shift
        y1 += shift
    if y1 > world_y_max:
        shift = y1 - world_y_max
        y0 -= shift
        y1 -= shift

    return (x0, x1, y0, y1)


def build_race_standings(
    entries: Sequence[tuple[str, tuple[int, int, int], float | None, float]],
    race_time: float,
) -> list[tuple[list[str], tuple[int, int, int]]]:
    """Build sidebar sections: standings header + per-racer place / gap / speed.

    Args:
        entries: ``(name, color, finish_time_or_None, speed_m_s)`` per racer.
        race_time: Elapsed race clock (s).

    Returns:
        Sidebar ``(lines, color)`` sections for :func:`draw_sidebar_panel`.
    """
    finished = [(n, c, float(t), s) for n, c, t, s in entries if t is not None]
    active = [(n, c, s) for n, c, t, s in entries if t is None]
    finished.sort(key=lambda row: row[2])

    ordered: list[tuple[str, tuple[int, int, int], float | None, float]] = []
    for name, color, finish, speed in finished:
        ordered.append((name, color, finish, speed))
    for name, color, speed in active:
        ordered.append((name, color, None, speed))

    from arco.config.palette import ui_rgb

    leader_time = finished[0][2] if finished else None
    header_color = ui_rgb("chrome_title")
    sections: list[tuple[list[str], tuple[int, int, int]]] = [
        (
            [
                "STANDINGS",
                f"  clock  {race_time:.1f} s",
            ],
            header_color,
        )
    ]
    for place, (name, color, finish, speed) in enumerate(ordered, start=1):
        if finish is not None:
            gap = ""
            if leader_time is not None and place > 1:
                gap = f"  (+{finish - leader_time:.1f})"
            status = f"GOAL {finish:.1f}s{gap}"
        else:
            status = f"racing · {speed:.1f} m/s"
        sections.append(
            (
                [
                    f"{place}  {name}",
                    f"  {status}",
                ],
                color,
            )
        )
    return sections


def make_minimap_surface(
    *,
    world_bounds: tuple[float, float, float, float],
    view_bounds: tuple[float, float, float, float],
    markers: Sequence[tuple[float, float, tuple[int, int, int]]],
    start: tuple[float, float],
    goal: tuple[float, float],
    size_px: int = MINIMAP_SIZE_PX,
) -> "pygame.Surface":
    """Render a dark inset map of the full city with view frame and racers.

    Args:
        world_bounds: Full map ``(x_min, x_max, y_min, y_max)``.
        view_bounds: Current follow-cam window (same tuple shape).
        markers: ``(x, y, rgb)`` racer dots.
        start: Start marker world position.
        goal: Goal marker world position.
        size_px: Square minimap edge length in pixels.

    Returns:
        A ``pygame.Surface`` (SRCALPHA) ready for :func:`blit_overlay`.
    """
    import pygame

    from arco.config.palette import ui_rgb

    wx0, wx1, wy0, wy1 = world_bounds
    world_w = max(wx1 - wx0, 1e-6)
    world_h = max(wy1 - wy0, 1e-6)
    pad = 8
    inner = max(size_px - 2 * pad, 8)

    bar = ui_rgb("chrome_bar")
    border = ui_rgb("chrome_border")
    mark = ui_rgb("chrome_title")
    surf = pygame.Surface((size_px, size_px), pygame.SRCALPHA)
    surf.fill((bar[0], bar[1], bar[2], 210))
    pygame.draw.rect(
        surf,
        (border[0], border[1], border[2], 255),
        pygame.Rect(0, 0, size_px, size_px),
        1,
    )

    def _to_px(x: float, y: float) -> tuple[int, int]:
        u = (x - wx0) / world_w
        v = (y - wy0) / world_h
        # pygame y grows downward; world y grows upward.
        return (
            pad + int(u * inner),
            pad + int((1.0 - v) * inner),
        )

    # Follow-cam frame on the full map.
    vx0, vx1, vy0, vy1 = view_bounds
    corners = [
        _to_px(vx0, vy0),
        _to_px(vx1, vy0),
        _to_px(vx1, vy1),
        _to_px(vx0, vy1),
    ]
    pygame.draw.polygon(surf, (mark[0], mark[1], mark[2], 90), corners, 1)

    sx, sy = _to_px(start[0], start[1])
    gx, gy = _to_px(goal[0], goal[1])
    pygame.draw.circle(surf, mark, (sx, sy), 3)
    pygame.draw.circle(surf, mark, (gx, gy), 3, 1)

    for x, y, color in markers:
        px, py = _to_px(x, y)
        pygame.draw.circle(surf, color, (px, py), 4)

    return surf


def phase_chrome_title(phase: str) -> str:
    """Return the header title for the current city presentation phase.

    Deprecated alias for :func:`scenario_phase_title` — kept for tests.

    Args:
        phase: ``background``, ``racing``, or ``done``.

    Returns:
        Short title string for the chrome header.
    """
    from arco.simulator.sim.layout import scenario_phase_title

    return scenario_phase_title("city", phase)
