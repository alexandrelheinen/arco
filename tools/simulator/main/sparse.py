#!/usr/bin/env python
"""RRT* vs SST race on a cul-de-sac obstacle map.

Both planners compete on the same sparse environment featuring a U-shaped
concave obstacle that blocks the direct path to the goal.  Their exploration
trees are revealed simultaneously.  Once both paths are drawn, both vehicles
launch at the same instant — the first one to reach the goal wins.

The simulation stops 2 seconds after the last vehicle arrives.

Keyboard controls
-----------------
SPACE         Pause / resume
R             Restart from the beginning
Q / Escape    Quit

Usage
-----
::

    cd tools/simulator
    python main/sparse.py

Optional flags::

    python main/sparse.py --fps 30
    python main/sparse.py --record /tmp/race.mp4 --record-duration 90
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys

# Make arco and tools packages importable without a full install.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_HERE, ".."))

import pygame
from renderer import WorldTransform, draw_trajectory
from scenes.sparse import SparseScene
from sim.tracking import VehicleConfig, build_vehicle_sim, find_lookahead
from sim.video import VideoWriter

from config import load_config

logger = logging.getLogger(__name__)

_DEFAULT_SCREEN_W = 1280
_DEFAULT_SCREEN_H = 800

# Frames to hold both completed trees before starting the race.
_HOLD_FRAMES = 60
# Simulation seconds to keep running after the last vehicle reaches the goal.
_POST_FINISH_SECS = 2.0

# ---------------------------------------------------------------------------
# Colour constants — vehicle body / arrow / trajectory / HUD text
# ---------------------------------------------------------------------------
_C_RRT_VEH: tuple[int, int, int] = (100, 160, 255)
_C_RRT_ARROW: tuple[int, int, int] = (180, 210, 255)
_C_RRT_TRAJ: tuple[int, int, int] = (60, 120, 200)
_C_RRT_HUD: tuple[int, int, int] = (130, 190, 255)

_C_SST_VEH: tuple[int, int, int] = (60, 235, 210)
_C_SST_ARROW: tuple[int, int, int] = (160, 255, 235)
_C_SST_TRAJ: tuple[int, int, int] = (40, 180, 155)
_C_SST_HUD: tuple[int, int, int] = (100, 240, 210)

_C_HUD: tuple[int, int, int] = (220, 220, 220)
_C_HUD_DIM: tuple[int, int, int] = (130, 130, 130)
_C_HUD_SHADOW: tuple[int, int, int] = (40, 40, 50)
_C_WINNER: tuple[int, int, int] = (255, 215, 50)
_C_TIE: tuple[int, int, int] = (200, 200, 80)

# Vehicle sprite dimensions (pixels)
_VEH_L = 14
_VEH_W = 7


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _draw_vehicle(
    surface: pygame.Surface,
    x: float,
    y: float,
    heading: float,
    transform: object,
    body: tuple[int, int, int],
    arrow: tuple[int, int, int],
) -> None:
    """Draw a coloured oriented vehicle rectangle at world position (x, y).

    Args:
        surface: Pygame surface to draw onto.
        x: Vehicle x-position in world metres.
        y: Vehicle y-position in world metres.
        heading: Vehicle heading in radians (0 = east, π/2 = north).
        transform: World-to-screen callable.
        body: RGB fill colour.
        arrow: RGB outline and arrow colour.
    """
    cx, cy = transform(x, y)
    ch = math.cos(heading)
    sh = math.sin(heading)
    hl, hw = _VEH_L / 2, _VEH_W / 2
    corners = []
    for lx, ly in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)):
        rx = lx * ch - ly * sh
        ry = -(lx * sh + ly * ch)
        corners.append((cx + rx, cy + ry))
    pygame.draw.polygon(surface, body, corners)
    pygame.draw.polygon(surface, arrow, corners, 1)
    tip = (cx + _VEH_L * 0.7 * ch, cy - _VEH_L * 0.7 * sh)
    pygame.draw.line(surface, arrow, (cx, cy), tip, 2)


def _blit_left(
    surface: pygame.Surface,
    font: pygame.font.Font,
    lines: list[str],
    color: tuple[int, int, int],
    x: int = 10,
    y: int = 10,
) -> None:
    """Render left-aligned shadowed text lines starting at (x, y).

    Args:
        surface: Pygame surface.
        font: Pygame font.
        lines: Text lines to render top-to-bottom.
        color: RGB text colour.
        x: Left x pixel position.
        y: Starting y pixel position.
    """
    for line in lines:
        shadow = font.render(line, True, _C_HUD_SHADOW)
        surface.blit(shadow, (x + 1, y + 1))
        rendered = font.render(line, True, color)
        surface.blit(rendered, (x, y))
        y += font.get_linesize() + 2


def _blit_right(
    surface: pygame.Surface,
    font: pygame.font.Font,
    lines: list[str],
    color: tuple[int, int, int],
    x_margin: int = 10,
    y: int = 10,
) -> None:
    """Render right-aligned shadowed text lines.

    Args:
        surface: Pygame surface.
        font: Pygame font.
        lines: Text lines to render top-to-bottom.
        color: RGB text colour.
        x_margin: Gap between text right edge and screen right edge.
        y: Starting y pixel position.
    """
    for line in lines:
        rendered = font.render(line, True, color)
        x = surface.get_width() - x_margin - rendered.get_width()
        shadow = font.render(line, True, _C_HUD_SHADOW)
        surface.blit(shadow, (x + 1, y + 1))
        surface.blit(rendered, (x, y))
        y += font.get_linesize() + 2


def _blit_center(
    surface: pygame.Surface,
    font: pygame.font.Font,
    line: str,
    color: tuple[int, int, int],
    y: int,
) -> None:
    """Render a single centred line at vertical position y.

    Args:
        surface: Pygame surface.
        font: Pygame font.
        line: Text to render.
        color: RGB text colour.
        y: Vertical pixel position.
    """
    rendered = font.render(line, True, color)
    x = (surface.get_width() - rendered.get_width()) // 2
    shadow = font.render(line, True, _C_HUD_SHADOW)
    surface.blit(shadow, (x + 1, y + 1))
    surface.blit(rendered, (x, y))


def _draw_winner_banner(
    surface: pygame.Surface,
    big_font: pygame.font.Font,
    text: str,
    color: tuple[int, int, int],
) -> None:
    """Draw a translucent centred banner with large winner text.

    Args:
        surface: Pygame surface.
        big_font: Large pygame font.
        text: Banner text (e.g. ``"RRT* WINS!"``).
        color: RGB text colour.
    """
    rendered = big_font.render(text, True, color)
    rw, rh = rendered.get_width(), rendered.get_height()
    pad = 14
    bx = (surface.get_width() - rw) // 2 - pad
    by = (surface.get_height() - rh) // 2 - pad
    banner = pygame.Surface((rw + 2 * pad, rh + 2 * pad), pygame.SRCALPHA)
    banner.fill((10, 10, 20, 200))
    surface.blit(banner, (bx, by))
    surface.blit(rendered, (bx + pad, by + pad))


# ---------------------------------------------------------------------------
# Race simulation
# ---------------------------------------------------------------------------


def run_race(
    scene: SparseScene,
    *,
    fps: int = 30,
    dt: float = 0.1,
    record: str = "",
    record_duration: float = 90.0,
) -> None:
    """Run the dual-vehicle cul-de-sac race.

    Phase 1 — **planning reveal**: both exploration trees grow on screen
    simultaneously.  The race does not start until both trees are fully drawn.

    Phase 2 — **racing**: both vehicles follow their respective planned paths
    from a shared start.  The first to arrive is declared the winner.  The
    simulation continues for :data:`_POST_FINISH_SECS` after the last vehicle
    reaches the goal, then exits.

    Args:
        scene: Fully built :class:`~scenes.sparse.SparseScene`.
        fps: Target frame rate in frames per second.
        dt: Simulation timestep in seconds.
        record: Output MP4 file path.  Empty string means interactive mode.
        record_duration: Maximum headless recording length in seconds.
    """
    recording = bool(record)
    max_record_frames = int(fps * record_duration)

    if recording:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    pygame.init()

    if recording:
        sw, sh = _DEFAULT_SCREEN_W, _DEFAULT_SCREEN_H
    else:
        info = pygame.display.Info()
        w = int(getattr(info, "current_w", 0) or 0)
        h = int(getattr(info, "current_h", 0) or 0)
        if w > 0 and h > 0:
            sw, sh = max(640, int(w * 0.9)), max(480, int(h * 0.9))
        else:
            sw, sh = _DEFAULT_SCREEN_W, _DEFAULT_SCREEN_H

    screen_size = (sw, sh)
    screen = pygame.display.set_mode(screen_size)
    clock = pygame.time.Clock()

    # Fonts — build scene after pygame.init so SysFont is safe.
    scene.build()
    pygame.display.set_caption(scene.title)

    font = pygame.font.SysFont("monospace", 14)
    big_font = pygame.font.SysFont("monospace", 36, bold=True)

    # World-to-screen transform (full view, no follow camera for a race).
    transform = WorldTransform(scene.world_points, screen_size, margin=60)

    cfg = scene.vehicle_config
    rrt_wps = scene.rrt_waypoints
    sst_wps = scene.sst_waypoints
    rrt_total = scene.rrt_total
    sst_total = scene.sst_total

    # Pacing: reveal both trees in parallel, finishing together at ~half-time.
    half_frames = (
        max(1, max_record_frames // 2) if recording else max(1, fps * 8)
    )
    nodes_per_frame = max(1, max(rrt_total, sst_total) // half_frames)

    # ---------------------------------------------------------------------------
    # Mutable simulation state
    # ---------------------------------------------------------------------------
    phase = "background"  # "background" | "racing" | "done"

    # Background-reveal counters
    rrt_revealed = 0
    sst_revealed = 0
    hold = 0

    # Vehicle state — initialised when the racing phase begins
    rrt_vehicle = None
    sst_vehicle = None
    rrt_loop = None
    sst_loop = None
    rrt_traj: list[tuple[float, float, float]] = []
    sst_traj: list[tuple[float, float, float]] = []
    rrt_finished = False
    sst_finished = False
    rrt_finish_time: float | None = None
    sst_finish_time: float | None = None
    last_finish_time: float | None = None
    race_time = 0.0

    paused = False

    def _start_racing() -> None:
        nonlocal rrt_vehicle, rrt_loop, rrt_traj
        nonlocal sst_vehicle, sst_loop, sst_traj
        nonlocal rrt_finished, sst_finished, race_time
        rrt_traj = []
        sst_traj = []
        rrt_finished = False
        sst_finished = False
        race_time = 0.0
        if rrt_wps:
            rrt_vehicle, rrt_loop = build_vehicle_sim(rrt_wps, cfg)
        if sst_wps:
            sst_vehicle, sst_loop = build_vehicle_sim(sst_wps, cfg)
        logger.info("Race started.")

    def _restart() -> None:
        nonlocal phase, rrt_revealed, sst_revealed, hold
        nonlocal rrt_finish_time, sst_finish_time, last_finish_time, paused
        phase = "background"
        rrt_revealed = 0
        sst_revealed = 0
        hold = 0
        rrt_finish_time = None
        sst_finish_time = None
        last_finish_time = None
        scene._sdf_surface = None  # Force SDF rebake on next draw
        paused = False

    # ---------------------------------------------------------------------------
    # Video writer
    # ---------------------------------------------------------------------------
    writer: VideoWriter | None = (
        VideoWriter(record, sw, sh, fps) if recording else None
    )
    if writer is not None:
        writer.open()

    record_frames = 0
    running = True

    try:
        while running:
            # ------------------------------------------------------------------
            # Event handling (interactive mode only)
            # ------------------------------------------------------------------
            if not recording:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_q, pygame.K_ESCAPE):
                            running = False
                        elif event.key == pygame.K_SPACE:
                            paused = not paused
                        elif event.key == pygame.K_r:
                            _restart()

            # ------------------------------------------------------------------
            # Phase logic
            # ------------------------------------------------------------------
            if not paused:
                if phase == "background":
                    rrt_revealed = min(
                        rrt_revealed + nodes_per_frame, rrt_total
                    )
                    sst_revealed = min(
                        sst_revealed + nodes_per_frame, sst_total
                    )
                    both_done = (
                        rrt_revealed >= rrt_total and sst_revealed >= sst_total
                    )
                    if both_done:
                        hold += 1
                        if hold >= _HOLD_FRAMES or (
                            recording and record_frames >= half_frames
                        ):
                            phase = "racing"
                            _start_racing()
                            logger.info("Switched to racing phase.")

                elif phase == "racing":
                    race_time += dt

                    # Advance RRT* vehicle
                    if rrt_vehicle is not None and rrt_loop is not None:
                        if not rrt_finished:
                            rrt_loop.step(rrt_wps, dt=dt)
                            rrt_traj.append(rrt_vehicle.pose)
                            gx, gy = rrt_wps[-1]
                            if (
                                math.hypot(
                                    rrt_vehicle.x - gx, rrt_vehicle.y - gy
                                )
                                < cfg.goal_radius
                            ):
                                rrt_finished = True
                                rrt_finish_time = race_time
                                logger.info(
                                    "RRT* reached goal at t=%.2f s", race_time
                                )

                    # Advance SST vehicle
                    if sst_vehicle is not None and sst_loop is not None:
                        if not sst_finished:
                            sst_loop.step(sst_wps, dt=dt)
                            sst_traj.append(sst_vehicle.pose)
                            gx, gy = sst_wps[-1]
                            if (
                                math.hypot(
                                    sst_vehicle.x - gx, sst_vehicle.y - gy
                                )
                                < cfg.goal_radius
                            ):
                                sst_finished = True
                                sst_finish_time = race_time
                                logger.info(
                                    "SST reached goal at t=%.2f s", race_time
                                )

                    # Update last-finish timestamp
                    if rrt_finished or sst_finished:
                        candidate = max(
                            t
                            for t in (rrt_finish_time, sst_finish_time)
                            if t is not None
                        )
                        last_finish_time = candidate

                    # Stop 2 s after the last vehicle arrives
                    if (
                        last_finish_time is not None
                        and race_time - last_finish_time >= _POST_FINISH_SECS
                    ):
                        phase = "done"
                        logger.info("Race finished.  Exiting.")

                elif phase == "done":
                    running = False

            # ------------------------------------------------------------------
            # Render
            # ------------------------------------------------------------------
            screen.fill(scene.bg_color)

            scene.draw_background(
                screen, transform, rrt_revealed, sst_revealed
            )

            if phase == "background":
                _draw_planning_hud(
                    screen,
                    font,
                    rrt_revealed,
                    rrt_total,
                    scene._rrt_path is not None,
                    sst_revealed,
                    sst_total,
                    scene._sst_path is not None,
                    paused,
                )

            elif phase in ("racing", "done"):
                # Trajectories
                if len(rrt_traj) >= 2:
                    _draw_colored_trajectory(
                        screen, rrt_traj, transform, _C_RRT_TRAJ
                    )
                if len(sst_traj) >= 2:
                    _draw_colored_trajectory(
                        screen, sst_traj, transform, _C_SST_TRAJ
                    )

                # Lookahead targets
                if rrt_vehicle is not None and not rrt_finished:
                    la = find_lookahead(
                        rrt_vehicle.x,
                        rrt_vehicle.y,
                        rrt_wps,
                        cfg.lookahead_distance,
                    )
                    _draw_lookahead(screen, la, transform, _C_RRT_VEH)
                if sst_vehicle is not None and not sst_finished:
                    la = find_lookahead(
                        sst_vehicle.x,
                        sst_vehicle.y,
                        sst_wps,
                        cfg.lookahead_distance,
                    )
                    _draw_lookahead(screen, la, transform, _C_SST_VEH)

                # Vehicles
                if rrt_vehicle is not None:
                    _draw_vehicle(
                        screen,
                        rrt_vehicle.x,
                        rrt_vehicle.y,
                        rrt_vehicle.heading,
                        transform,
                        _C_RRT_VEH,
                        _C_RRT_ARROW,
                    )
                if sst_vehicle is not None:
                    _draw_vehicle(
                        screen,
                        sst_vehicle.x,
                        sst_vehicle.y,
                        sst_vehicle.heading,
                        transform,
                        _C_SST_VEH,
                        _C_SST_ARROW,
                    )

                _draw_race_hud(
                    screen,
                    font,
                    race_time,
                    rrt_finish_time,
                    sst_finish_time,
                    paused,
                )

                # Winner / tie banner (shown once first vehicle finishes)
                if rrt_finished or sst_finished:
                    both = rrt_finished and sst_finished
                    if both:
                        diff = abs(
                            (rrt_finish_time or 0.0) - (sst_finish_time or 0.0)
                        )
                        if diff < 0.15:
                            _draw_winner_banner(
                                screen, big_font, "IT'S A TIE!", _C_TIE
                            )
                        elif (rrt_finish_time or math.inf) < (
                            sst_finish_time or math.inf
                        ):
                            _draw_winner_banner(
                                screen, big_font, "RRT*  WINS!", _C_RRT_HUD
                            )
                        else:
                            _draw_winner_banner(
                                screen, big_font, "SST  WINS!", _C_SST_HUD
                            )
                    elif rrt_finished:
                        _draw_winner_banner(
                            screen, big_font, "RRT*  LEADS!", _C_RRT_HUD
                        )
                    else:
                        _draw_winner_banner(
                            screen, big_font, "SST  LEADS!", _C_SST_HUD
                        )

            # ------------------------------------------------------------------
            # Output frame
            # ------------------------------------------------------------------
            if recording and writer is not None:
                writer.write_frame(screen)
                record_frames += 1
                if record_frames >= max_record_frames:
                    running = False
                elif phase == "done":
                    running = False
            else:
                pygame.display.flip()
                clock.tick(fps)

    finally:
        if writer is not None:
            writer.close()
        pygame.quit()


# ---------------------------------------------------------------------------
# Private rendering helpers (local to this module)
# ---------------------------------------------------------------------------


def _draw_colored_trajectory(
    surface: pygame.Surface,
    traj: list[tuple[float, float, float]],
    transform: object,
    color: tuple[int, int, int],
) -> None:
    """Draw a past-pose trajectory as a simple polyline in *color*.

    Args:
        surface: Pygame surface.
        traj: Ordered list of ``(x, y, heading)`` poses.
        transform: World-to-screen callable.
        color: Polyline RGB colour.
    """
    if len(traj) < 2:
        return
    pts = [transform(float(p[0]), float(p[1])) for p in traj]
    pygame.draw.lines(surface, color, False, pts, 1)


def _draw_lookahead(
    surface: pygame.Surface,
    target: tuple[float, float],
    transform: object,
    color: tuple[int, int, int],
) -> None:
    """Draw the pure-pursuit lookahead target as a small circle.

    Args:
        surface: Pygame surface.
        target: World ``(x, y)`` of the lookahead point.
        transform: World-to-screen callable.
        color: Circle colour.
    """
    sx, sy = transform(float(target[0]), float(target[1]))
    pygame.draw.circle(surface, color, (sx, sy), 5)
    pygame.draw.circle(surface, (28, 28, 35), (sx, sy), 2)


def _draw_planning_hud(
    surface: pygame.Surface,
    font: pygame.font.Font,
    rrt_revealed: int,
    rrt_total: int,
    rrt_found: bool,
    sst_revealed: int,
    sst_total: int,
    sst_found: bool,
    paused: bool,
) -> None:
    """Draw the planning-phase HUD with per-planner progress.

    RRT* info is shown top-left; SST info is shown top-right.

    Args:
        surface: Pygame surface.
        font: Pygame font.
        rrt_revealed: Nodes revealed so far for RRT*.
        rrt_total: Total RRT* nodes.
        rrt_found: Whether RRT* found a path.
        sst_revealed: Nodes revealed so far for SST.
        sst_total: Total SST nodes.
        sst_found: Whether SST found a path.
        paused: Whether simulation is paused.
    """
    rrt_lines = [
        "RRT*",
        f"Nodes: {rrt_revealed}/{rrt_total}",
        f"Path:  {'found' if rrt_found else 'none'}",
    ]
    sst_lines = [
        "SST",
        f"Nodes: {sst_revealed}/{sst_total}",
        f"Path:  {'found' if sst_found else 'none'}",
    ]
    _blit_left(surface, font, rrt_lines, _C_RRT_HUD)
    _blit_right(surface, font, sst_lines, _C_SST_HUD)

    both_ready = rrt_revealed >= rrt_total and sst_revealed >= sst_total
    center_line = (
        "[ PAUSED — press SPACE ]"
        if paused
        else (
            "Both paths ready — launching race…" if both_ready else "Planning…"
        )
    )
    _blit_center(surface, font, center_line, _C_HUD, surface.get_height() - 24)


def _draw_race_hud(
    surface: pygame.Surface,
    font: pygame.font.Font,
    race_time: float,
    rrt_finish: float | None,
    sst_finish: float | None,
    paused: bool,
) -> None:
    """Draw the racing-phase HUD showing per-vehicle status and race timer.

    Args:
        surface: Pygame surface.
        font: Pygame font.
        race_time: Elapsed race simulation time in seconds.
        rrt_finish: Simulation time at which RRT* vehicle finished, or None.
        sst_finish: Simulation time at which SST vehicle finished, or None.
        paused: Whether the simulation is paused.
    """
    if rrt_finish is None:
        rrt_status = f"t = {race_time:.1f} s"
    else:
        rrt_status = f"GOAL  in {rrt_finish:.1f} s"

    if sst_finish is None:
        sst_status = f"t = {race_time:.1f} s"
    else:
        sst_status = f"GOAL  in {sst_finish:.1f} s"

    _blit_left(surface, font, ["RRT*", rrt_status], _C_RRT_HUD)
    _blit_right(surface, font, ["SST", sst_status], _C_SST_HUD)

    center = f"Race  {race_time:.1f} s"
    if paused:
        center = "[ PAUSED — press SPACE ]"
    _blit_center(surface, font, center, _C_HUD, 10)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and launch the cul-de-sac race."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        metavar="N",
        help="Target frame rate (default: 30).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        metavar="S",
        help="Simulation timestep in seconds (default: 0.1).",
    )
    parser.add_argument(
        "--record",
        metavar="FILE",
        default="",
        help="Record a headless MP4 to FILE (requires ffmpeg).",
    )
    parser.add_argument(
        "--record-duration",
        type=float,
        default=90.0,
        metavar="S",
        dest="record_duration",
        help="Maximum recording duration in seconds (default: 90).",
    )
    args = parser.parse_args()

    scene = SparseScene(load_config("sparse"))
    run_race(
        scene,
        fps=args.fps,
        dt=args.dt,
        record=args.record,
        record_duration=args.record_duration,
    )


if __name__ == "__main__":
    main()
