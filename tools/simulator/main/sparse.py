#!/usr/bin/env python
"""RRT* vs SST race on a cul-de-sac obstacle map.

Both planners compete on the same sparse environment featuring a U-shaped
concave obstacle that blocks the direct path to the goal.  Their exploration
trees are revealed simultaneously.  Once both paths are drawn, both vehicles
launch at the same instant — the first one to reach the goal wins.

The simulation keeps the victory screen open until the user closes it.
Pass ``--close`` to exit automatically once the race is done.

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
    python main/sparse.py --close
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
import renderer_gl
from groundhog import make_groundhog_surface
from OpenGL.GL import (  # type: ignore[import-untyped]
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_LIGHTING,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_SMOOTH,
    GL_SRC_ALPHA,
    glBlendFunc,
    glClear,
    glClearColor,
    glDisable,
    glEnable,
    glShadeModel,
)
from scenes.sparse import SparseScene
from sim.loading import run_with_loading_screen
from sim.tracking import VehicleConfig, build_vehicle_sim, find_lookahead
from sim.video import VideoWriter

from config import load_config

logger = logging.getLogger(__name__)

_DEFAULT_SCREEN_W = 1280
_DEFAULT_SCREEN_H = 720

# Frames to hold both completed trees before starting the race.
_HOLD_FRAMES = 60
# Simulation seconds to keep running after the last vehicle reaches the goal.
_POST_FINISH_SECS = 2.0

# ---------------------------------------------------------------------------
# Color constants — loaded from tools/config/colors.yml
# ---------------------------------------------------------------------------
_COLORS = load_config("colors")


def _rgb(section: str, key: str) -> tuple[int, int, int]:
    v = _COLORS[section][key]
    return (int(v[0]), int(v[1]), int(v[2]))


_C_RRT_VEH: tuple[int, int, int] = _rgb("rrt", "vehicle")
_C_RRT_TRAJ: tuple[int, int, int] = _rgb("rrt", "trail")
_C_RRT_HUD: tuple[int, int, int] = _rgb("rrt", "hud")

_C_SST_VEH: tuple[int, int, int] = _rgb("sst", "vehicle")
_C_SST_TRAJ: tuple[int, int, int] = _rgb("sst", "trail")
_C_SST_HUD: tuple[int, int, int] = _rgb("sst", "hud")

_C_HUD: tuple[int, int, int] = _rgb("hud", "text")
_C_HUD_DIM: tuple[int, int, int] = _rgb("hud", "dim")
_C_HUD_SHADOW: tuple[int, int, int] = _rgb("hud", "shadow")
_C_WINNER: tuple[int, int, int] = _rgb("hud", "winner")
_C_TIE: tuple[int, int, int] = _rgb("hud", "tie")

# Vehicle body world dimensions
_VEH_HALF_L = 1.5  # meters
_VEH_HALF_W = 0.7  # meters
_LOOKAHEAD_DISC_R = 0.5  # meters


def _c(t: tuple[int, int, int]) -> tuple[float, float, float]:
    return (t[0] / 255.0, t[1] / 255.0, t[2] / 255.0)


# ---------------------------------------------------------------------------
# HUD helpers — build a pygame surface then blit_overlay
# ---------------------------------------------------------------------------


def _make_text_surface(
    font: pygame.font.Font,
    lines: list[str],
    color: tuple[int, int, int],
) -> pygame.Surface:
    """Build a small SRCALPHA surface with the given text lines.

    Args:
        font: Pygame monospace font.
        lines: Lines to render top-to-bottom.
        color: RGB text color.

    Returns:
        Transparent SRCALPHA pygame surface.
    """
    line_h = font.get_linesize() + 2
    panel_h = len(lines) * line_h + 8
    panel_w = max((font.size(ln)[0] for ln in lines), default=10) + 20
    surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    surf.fill((10, 10, 20, 180))
    y = 4
    for line in lines:
        shadow = font.render(line, True, _C_HUD_SHADOW)
        surf.blit(shadow, (11, y + 1))
        rendered = font.render(line, True, color)
        surf.blit(rendered, (10, y))
        y += line_h
    return surf


def _blit_left(
    font: pygame.font.Font,
    lines: list[str],
    color: tuple[int, int, int],
    sw: int,
    sh: int,
    x: int = 10,
    y: int = 10,
) -> None:
    """Render left-aligned shadowed text lines starting at (x, y).

    Args:
        font: Pygame font.
        lines: Text lines to render top-to-bottom.
        color: RGB text color.
        sw: Screen width in pixels.
        sh: Screen height in pixels.
        x: Left x pixel position.
        y: Starting y pixel position.
    """
    surf = _make_text_surface(font, lines, color)
    renderer_gl.blit_overlay(surf, x, y, sw, sh)


def _blit_right(
    font: pygame.font.Font,
    lines: list[str],
    color: tuple[int, int, int],
    sw: int,
    sh: int,
    x_margin: int = 10,
    y: int = 10,
) -> None:
    """Render right-aligned shadowed text lines.

    Args:
        font: Pygame font.
        lines: Text lines to render top-to-bottom.
        color: RGB text color.
        sw: Screen width in pixels.
        sh: Screen height in pixels.
        x_margin: Gap between text right edge and screen right edge.
        y: Starting y pixel position.
    """
    surf = _make_text_surface(font, lines, color)
    x = sw - x_margin - surf.get_width()
    renderer_gl.blit_overlay(surf, x, y, sw, sh)


def _blit_center(
    font: pygame.font.Font,
    line: str,
    color: tuple[int, int, int],
    sw: int,
    sh: int,
    y: int,
) -> None:
    """Render a single centered line at vertical position y.

    Args:
        font: Pygame font.
        line: Text to render.
        color: RGB text color.
        sw: Screen width in pixels.
        sh: Screen height in pixels.
        y: Vertical pixel position.
    """
    surf = _make_text_surface(font, [line], color)
    x = (sw - surf.get_width()) // 2
    renderer_gl.blit_overlay(surf, x, y, sw, sh)


def _draw_winner_banner(
    font: pygame.font.Font,
    text: str,
    color: tuple[int, int, int],
    sw: int,
    sh: int,
) -> None:
    """Draw a translucent centered banner with winner text and groundhog mascot.

    The groundhog is rendered in *color* and placed to the left of the winner
    text inside the banner, giving the victory screen a visual identity.

    Args:
        font: Large pygame font.
        text: Banner text (e.g. ``"RRT*  WINS!"``).
        color: RGB text color (also used for the groundhog fill).
        sw: Screen width in pixels.
        sh: Screen height in pixels.
    """
    rendered = font.render(text, True, color)
    rw, rh = rendered.get_width(), rendered.get_height()
    groundhog_surf = make_groundhog_surface(color, height=rh + 30)
    mw, mh = groundhog_surf.get_size()

    gap = 16
    pad = 14
    banner_w = pad + mw + gap + rw + pad
    banner_h = max(mh, rh) + 2 * pad

    banner = pygame.Surface((banner_w, banner_h), pygame.SRCALPHA)
    banner.fill((10, 10, 20, 200))
    banner.blit(groundhog_surf, (pad, (banner_h - mh) // 2))
    banner.blit(rendered, (pad + mw + gap, (banner_h - rh) // 2))

    bx = (sw - banner.get_width()) // 2
    by = (sh - banner.get_height()) // 2
    renderer_gl.blit_overlay(banner, bx, by, sw, sh)


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
    auto_close: bool = False,
) -> None:
    """Run the dual-vehicle cul-de-sac race.

    Phase 1 — **planning reveal**: both exploration trees grow on screen
    simultaneously.  The race does not start until both trees are fully drawn.

    Phase 2 — **racing**: both vehicles follow their respective planned paths
    from a shared start.  The first to arrive is declared the winner.  The
    simulation continues for :data:`_POST_FINISH_SECS` after the last vehicle
    reaches the goal, then transitions to **done**.

    Phase 3 — **done**: the victory screen stays visible.  In interactive
    mode the user presses **R** to restart or **Q/Escape** to quit.  Pass
    ``auto_close=True`` (or use ``--close``) to exit automatically.

    Args:
        scene: Fully built :class:`~scenes.sparse.SparseScene`.
        fps: Target frame rate in frames per second.
        dt: Simulation timestep in seconds.
        record: Output MP4 file path.  Empty string means interactive mode.
        record_duration: Maximum headless recording length in seconds.
        auto_close: If ``True``, exit automatically once the race is done
            (interactive mode only; recording mode always exits on done).
    """
    recording = bool(record)
    max_record_frames = int(fps * record_duration)

    # OpenGL requires a real (or virtual) display.  For headless recording
    # use xvfb-run.
    if recording:
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
    pygame.display.set_mode(screen_size, pygame.OPENGL | pygame.DOUBLEBUF)
    clock = pygame.time.Clock()

    # Fonts — build scene after pygame.init so SysFont is safe.
    run_with_loading_screen(scene, sw, sh, bg_color=scene.bg_color)
    pygame.display.set_caption(scene.title)

    font = pygame.font.SysFont("monospace", 14)
    big_font = pygame.font.SysFont("monospace", 36, bold=True)

    # GL state
    bg = scene.bg_color
    glClearColor(bg[0] / 255.0, bg[1] / 255.0, bg[2] / 255.0, 1.0)
    glShadeModel(GL_SMOOTH)
    glDisable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glDisable(GL_LIGHTING)

    # World bounds and projection (full fixed view — no follow camera for race).
    wpts = scene.world_points
    _all_x = [p[0] for p in wpts]
    _all_y = [p[1] for p in wpts]
    x_min, x_max = min(_all_x), max(_all_x)
    y_min, y_max = min(_all_y), max(_all_y)

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
    # Discrete background stages for LEFT/RIGHT navigation:
    # 0 = empty init, 1 = RRT* tree complete, 2 = both trees complete.
    _bg_stages = [(0, 0), (rrt_total, 0), (rrt_total, sst_total)]
    _bg_stage_idx = 0

    phase = "background"  # "background" | "racing" | "done"

    rrt_revealed = 0
    sst_revealed = 0
    hold = 0

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
        nonlocal _bg_stage_idx
        phase = "background"
        rrt_revealed = 0
        sst_revealed = 0
        hold = 0
        _bg_stage_idx = 0
        rrt_finish_time = None
        sst_finish_time = None
        last_finish_time = None
        scene._sdf_tex_id = None  # Force SDF rebake on next draw
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
                        elif (
                            event.key == pygame.K_RIGHT
                            and phase == "background"
                        ):
                            _bg_stage_idx = min(
                                len(_bg_stages) - 1, _bg_stage_idx + 1
                            )
                            rrt_revealed, sst_revealed = _bg_stages[
                                _bg_stage_idx
                            ]
                            if (
                                rrt_revealed < rrt_total
                                or sst_revealed < sst_total
                            ):
                                hold = 0
                        elif (
                            event.key == pygame.K_LEFT
                            and phase == "background"
                        ):
                            _bg_stage_idx = max(0, _bg_stage_idx - 1)
                            rrt_revealed, sst_revealed = _bg_stages[
                                _bg_stage_idx
                            ]
                            hold = 0

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

                    if rrt_finished or sst_finished:
                        candidate = max(
                            t
                            for t in (rrt_finish_time, sst_finish_time)
                            if t is not None
                        )
                        last_finish_time = candidate

                    if (
                        last_finish_time is not None
                        and race_time - last_finish_time >= _POST_FINISH_SECS
                    ):
                        phase = "done"
                        logger.info("Race finished.")

                elif phase == "done":
                    pass  # Keep alive; press R to restart or Q to quit.

            # ------------------------------------------------------------------
            # Render
            # ------------------------------------------------------------------
            bg = scene.bg_color
            glClearColor(bg[0] / 255.0, bg[1] / 255.0, bg[2] / 255.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT)

            renderer_gl.setup_2d_projection(x_min, x_max, y_min, y_max, sw, sh)

            scene.draw_background(
                rrt_revealed,
                sst_revealed,
                racing=(phase in ("racing", "done")),
            )

            if phase == "background":
                _draw_planning_hud(
                    font,
                    rrt_revealed,
                    rrt_total,
                    scene._rrt_path is not None,
                    sst_revealed,
                    sst_total,
                    scene._sst_path is not None,
                    paused,
                    sw,
                    sh,
                )

            elif phase in ("racing", "done"):
                if len(rrt_traj) >= 2:
                    renderer_gl.draw_path(
                        [(p[0], p[1]) for p in rrt_traj],
                        *_c(_C_RRT_TRAJ),
                        width=1.5,
                    )
                if len(sst_traj) >= 2:
                    renderer_gl.draw_path(
                        [(p[0], p[1]) for p in sst_traj],
                        *_c(_C_SST_TRAJ),
                        width=1.5,
                    )

                if rrt_vehicle is not None and not rrt_finished:
                    la = find_lookahead(
                        rrt_vehicle.x,
                        rrt_vehicle.y,
                        rrt_wps,
                        cfg.lookahead_distance,
                    )
                    renderer_gl.draw_disc(
                        la[0], la[1], _LOOKAHEAD_DISC_R, *_c(_C_RRT_VEH)
                    )
                if sst_vehicle is not None and not sst_finished:
                    la = find_lookahead(
                        sst_vehicle.x,
                        sst_vehicle.y,
                        sst_wps,
                        cfg.lookahead_distance,
                    )
                    renderer_gl.draw_disc(
                        la[0], la[1], _LOOKAHEAD_DISC_R, *_c(_C_SST_VEH)
                    )

                if rrt_vehicle is not None:
                    renderer_gl.draw_oriented_rect(
                        rrt_vehicle.x,
                        rrt_vehicle.y,
                        _VEH_HALF_L,
                        _VEH_HALF_W,
                        rrt_vehicle.heading,
                        *_c(_C_RRT_VEH),
                    )
                if sst_vehicle is not None:
                    renderer_gl.draw_oriented_rect(
                        sst_vehicle.x,
                        sst_vehicle.y,
                        _VEH_HALF_L,
                        _VEH_HALF_W,
                        sst_vehicle.heading,
                        *_c(_C_SST_VEH),
                    )

                _draw_race_hud(
                    font,
                    race_time,
                    rrt_finish_time,
                    sst_finish_time,
                    paused,
                    sw,
                    sh,
                )

                if rrt_finished or sst_finished:
                    both = rrt_finished and sst_finished
                    if both:
                        diff = abs(
                            (rrt_finish_time or 0.0) - (sst_finish_time or 0.0)
                        )
                        if diff < 0.15:
                            _draw_winner_banner(
                                big_font, "IT'S A TIE!", _C_TIE, sw, sh
                            )
                        elif (rrt_finish_time or math.inf) < (
                            sst_finish_time or math.inf
                        ):
                            _draw_winner_banner(
                                big_font, "RRT*  WINS!", _C_RRT_HUD, sw, sh
                            )
                        else:
                            _draw_winner_banner(
                                big_font, "SST  WINS!", _C_SST_HUD, sw, sh
                            )
                    elif rrt_finished:
                        _draw_winner_banner(
                            big_font, "RRT*  LEADS!", _C_RRT_HUD, sw, sh
                        )
                    else:
                        _draw_winner_banner(
                            big_font, "SST  LEADS!", _C_SST_HUD, sw, sh
                        )

            if phase == "done" and not recording:
                hint = (
                    "Press  R  to restart   |   Q  to quit"
                    if not auto_close
                    else "Press  Q  to quit"
                )
                _blit_center(
                    font,
                    hint,
                    _C_HUD_DIM,
                    sw,
                    sh,
                    sh - 34,
                )

            # ------------------------------------------------------------------
            # Output frame
            # ------------------------------------------------------------------
            if recording and writer is not None:
                pygame.display.flip()
                writer.write_frame_gl()
                record_frames += 1
                if record_frames >= max_record_frames:
                    running = False
                elif phase == "done":
                    running = False
            else:
                pygame.display.flip()
                clock.tick(fps)
                if auto_close and phase == "done":
                    running = False

    finally:
        if writer is not None:
            writer.close()
        pygame.quit()


# ---------------------------------------------------------------------------
# Private rendering helpers
# ---------------------------------------------------------------------------


def _draw_planning_hud(
    font: pygame.font.Font,
    rrt_revealed: int,
    rrt_total: int,
    rrt_found: bool,
    sst_revealed: int,
    sst_total: int,
    sst_found: bool,
    paused: bool,
    sw: int,
    sh: int,
) -> None:
    """Draw the planning-phase HUD with per-planner progress.

    RRT* info is shown top-left; SST info is shown top-right.

    Args:
        font: Pygame font.
        rrt_revealed: Nodes revealed so far for RRT*.
        rrt_total: Total RRT* nodes.
        rrt_found: Whether RRT* found a path.
        sst_revealed: Nodes revealed so far for SST.
        sst_total: Total SST nodes.
        sst_found: Whether SST found a path.
        paused: Whether simulation is paused.
        sw: Screen width in pixels.
        sh: Screen height in pixels.
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
    _blit_left(font, rrt_lines, _C_RRT_HUD, sw, sh)
    _blit_right(font, sst_lines, _C_SST_HUD, sw, sh)

    both_ready = rrt_revealed >= rrt_total and sst_revealed >= sst_total
    center_line = (
        "[ PAUSED — press SPACE ]"
        if paused
        else (
            "Both paths ready — launching race…" if both_ready else "Planning…"
        )
    )
    _blit_center(font, center_line, _C_HUD, sw, sh, sh - 34)


def _draw_race_hud(
    font: pygame.font.Font,
    race_time: float,
    rrt_finish: float | None,
    sst_finish: float | None,
    paused: bool,
    sw: int,
    sh: int,
) -> None:
    """Draw the racing-phase HUD showing per-vehicle status and race timer.

    Args:
        font: Pygame font.
        race_time: Elapsed race simulation time in seconds.
        rrt_finish: Simulation time at which RRT* vehicle finished, or None.
        sst_finish: Simulation time at which SST vehicle finished, or None.
        paused: Whether the simulation is paused.
        sw: Screen width in pixels.
        sh: Screen height in pixels.
    """
    if rrt_finish is None:
        rrt_status = f"t = {race_time:.1f} s"
    else:
        rrt_status = f"GOAL  in {rrt_finish:.1f} s"

    if sst_finish is None:
        sst_status = f"t = {race_time:.1f} s"
    else:
        sst_status = f"GOAL  in {sst_finish:.1f} s"

    _blit_left(font, ["RRT*", rrt_status], _C_RRT_HUD, sw, sh)
    _blit_right(font, ["SST", sst_status], _C_SST_HUD, sw, sh)

    center = f"Race  {race_time:.1f} s"
    if paused:
        center = "[ PAUSED — press SPACE ]"
    _blit_center(font, center, _C_HUD, sw, sh, 10)


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
    parser.add_argument(
        "--close",
        action="store_true",
        default=False,
        help="Close the window automatically when the race is done.",
    )
    args = parser.parse_args()

    scene = SparseScene(load_config("sparse"), load_config("obstacles"))
    run_race(
        scene,
        fps=args.fps,
        dt=args.dt,
        record=args.record,
        record_duration=args.record_duration,
        auto_close=args.close,
    )


if __name__ == "__main__":
    main()
