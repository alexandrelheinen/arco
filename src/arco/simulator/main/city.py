#!/usr/bin/env python
"""RRT* vs SST vs A* race on a cul-de-sac obstacle map.

    Three planners compete on the same sparse environment featuring a U-shaped
    concave obstacle that blocks the direct path to the goal. Their exploration
    trees are revealed simultaneously. Once all paths are drawn, all vehicles
    launch at the same instant — the first one to reach the goal wins.

The simulation stops 3 seconds after the second vehicle arrives.

Keyboard controls
-----------------
SPACE         Pause / resume
R             Restart from the beginning
Q / Escape    Quit

Usage
-----
::

    cd tools/simulator
    python main/city.py

Optional flags::

    python main/city.py --fps 30
    python main/city.py --record /tmp/race.mp4 --record-duration 90
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

import pygame
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

from arco.config import load_config
from arco.config.palette import layer_rgb, ui_rgb
from arco.simulator import renderer_gl
from arco.simulator.scenes import RaceScene
from arco.simulator.scenes.sparse import CityScene
from arco.simulator.sim.city_race_style import (
    DEFAULT_CITY_HORIZON_DT,
    DEFAULT_CITY_HORIZON_STEP_COUNT,
    LOOKAHEAD_DISC_R,
    PAST_TRACE_WIDTH,
    PREDICTED_TRACE_WIDTH,
    VEH_HALF_L,
    VEH_HALF_W,
)
from arco.simulator.sim.layout import (
    ScreenLayout,
    draw_sidebar_panel,
    make_chrome_surface,
)
from arco.simulator.sim.loading import run_with_loading_screen
from arco.simulator.sim.tracking import (
    VehicleConfig,
    build_vehicle_mpc_sim,
    build_vehicle_sim,
    find_lookahead,
    path_following_mpc_config_from_simulator,
)
from arco.simulator.sim.video import VideoWriter

logger = logging.getLogger(__name__)

_DEFAULT_SCREEN_W = 1280
_DEFAULT_SCREEN_H = 720

# Frames to hold both completed trees before starting the race.
_HOLD_FRAMES = 60
# Simulation seconds to keep running after the second vehicle reaches the goal.
_POST_FINISH_SECS = 3.0

# ---------------------------------------------------------------------------
# Color constants — derived from the unified palette module.
# All colors read from src/arco/config/colors.yml via palette.py; no
# hardcoded hex values or direct load_config("colors") calls here.
# ---------------------------------------------------------------------------
_C_RRT_VEH: tuple[int, int, int] = layer_rgb("rrt", "vehicle")
_C_RRT_TRAJ: tuple[int, int, int] = layer_rgb("rrt", "trajectory")
_C_RRT_HUD: tuple[int, int, int] = layer_rgb("rrt", "vehicle")

_C_SST_VEH: tuple[int, int, int] = layer_rgb("sst", "vehicle")
_C_SST_TRAJ: tuple[int, int, int] = layer_rgb("sst", "trajectory")
_C_SST_HUD: tuple[int, int, int] = layer_rgb("sst", "vehicle")

_C_ASTAR_VEH: tuple[int, int, int] = layer_rgb("astar", "vehicle")
_C_ASTAR_TRAJ: tuple[int, int, int] = layer_rgb("astar", "trajectory")
_C_ASTAR_HUD: tuple[int, int, int] = layer_rgb("astar", "vehicle")

_C_WINNER: tuple[int, int, int] = ui_rgb("hud_winner")
_C_TIE: tuple[int, int, int] = ui_rgb("hud_tie")


def _c(t: tuple[int, int, int]) -> tuple[float, float, float]:
    return (t[0] / 255.0, t[1] / 255.0, t[2] / 255.0)


def _format_clock(seconds: float) -> str:
    """Format seconds as ``MMminSSs`` rounded to whole seconds."""
    rounded = int(round(seconds))
    mins, secs = divmod(rounded, 60)
    return f"{mins:02d}min{secs:02d}s"


# ---------------------------------------------------------------------------
# Private rendering helpers
# ---------------------------------------------------------------------------


def _brighten_rgb(
    color: tuple[int, int, int], *, mix: float = 0.55
) -> tuple[float, float, float]:
    """Mix an RGB color toward white for a brighter on-dark fill.

    Args:
        color: Base RGB color in ``[0, 255]``.
        mix: Fraction of white blended in (``0`` = original, ``1`` = white).

    Returns:
        Brightened RGB components in ``[0, 1]``.
    """
    r, g, b = _c(color)
    m = float(min(max(mix, 0.0), 1.0))
    return (r + (1.0 - r) * m, g + (1.0 - g) * m, b + (1.0 - b) * m)


def _draw_race_vehicle(
    x: float,
    y: float,
    heading: float,
    color: tuple[int, int, int],
) -> None:
    """Draw a race agent as a bright oriented rectangle (not a disc).

    Args:
        x: Vehicle x position in world meters.
        y: Vehicle y position in world meters.
        heading: Vehicle heading in radians.
        color: Base RGB body color in ``[0, 255]``.
    """
    renderer_gl.draw_oriented_rect(
        x,
        y,
        VEH_HALF_L,
        VEH_HALF_W,
        heading,
        *_brighten_rgb(color, mix=0.65),
    )


def _draw_winner_banner(
    font: pygame.font.Font,
    text: str,
    color: tuple[int, int, int],
    sw: int,
    sh: int,
    layout: ScreenLayout | None = None,
) -> None:
    """Draw a translucent centered banner with large winner text.

    Args:
        font: Large pygame font.
        text: Banner text (e.g. ``"RRT* WINS!"``).
        color: RGB text color.
        sw: Screen width in pixels.
        sh: Screen height in pixels.
        layout: Optional layout for centering within the content area.
    """
    rendered = font.render(text, True, color)
    rw, rh = rendered.get_width(), rendered.get_height()
    pad = 14
    banner = pygame.Surface((rw + 2 * pad, rh + 2 * pad), pygame.SRCALPHA)
    banner.fill((10, 10, 20, 200))
    banner.blit(rendered, (pad, pad))
    if layout is not None:
        bx = layout.content_x + (layout.content_w - banner.get_width()) // 2
        by = layout.header_h + (layout.content_h - banner.get_height()) // 2
    else:
        bx = (sw - banner.get_width()) // 2
        by = (sh - banner.get_height()) // 2
    renderer_gl.blit_overlay(banner, bx, by, sw, sh)


# ---------------------------------------------------------------------------
# Race simulation
# ---------------------------------------------------------------------------


def run_race(
    scene: RaceScene,
    *,
    fps: int = 30,
    dt: float = 0.1,
    record: str = "",
    record_duration: float = 90.0,
    fast_record: bool = False,
) -> None:
    """Run the three-vehicle cul-de-sac race.

    Phase 1 — **planning reveal**: all exploration trees grow on screen
    simultaneously. The race does not start until every tree is fully drawn.

    Phase 2 — **racing**: all vehicles follow their respective planned paths
    from a shared start.  The first to arrive is declared the winner.  The
    simulation continues for :data:`_POST_FINISH_SECS` after the last
    vehicle reaches the goal, then exits.

    Args:
        scene: Fully built city race scene.
        fps: Target frame rate in frames per second.
        dt: Simulation timestep in seconds.
        record: Output MP4 file path.  Empty string means interactive mode.
        record_duration: Maximum headless recording length in seconds.
        fast_record: When recording, skip animated tree reveal and start the
            race immediately so the duration budget is spent on tracking.
    """
    recording = bool(record)
    max_record_frames = int(fps * record_duration)
    # Also honor YAML ``simulator.fast_record`` when the caller did not pass
    # the flag explicitly (arcosim --fast-record mutates the scene config).
    sim_cfg_early = getattr(scene, "_sim_cfg", None)
    if not isinstance(sim_cfg_early, dict):
        sim_cfg_early = (getattr(scene, "_cfg", {}) or {}).get("simulator", {})
    if not isinstance(sim_cfg_early, dict):
        sim_cfg_early = {}
    fast_record = bool(fast_record) or (
        recording and bool(sim_cfg_early.get("fast_record", False))
    )

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
    title_font = pygame.font.SysFont("monospace", 14, bold=True)
    big_font = pygame.font.SysFont("monospace", 36, bold=True)
    layout = ScreenLayout(sw, sh)

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
    # A* is optional — VehicleScene only has RRT* and SST.
    astar_wps: list[tuple[float, float]] = getattr(
        scene, "astar_waypoints", []
    )
    rrt_total = scene.rrt_total
    sst_total = scene.sst_total
    astar_total: int = getattr(scene, "astar_total", 0)
    rrt_metrics = scene.rrt_metrics
    sst_metrics = scene.sst_metrics
    default_astar_metrics: dict[str, Any] = {
        "steps": 0,
        "nodes": 0,
        "planner_time": 0.0,
        "planned_path_length": 0.0,
        "trajectory_arc_length": 0.0,
        "trajectory_duration": 0.0,
        "path_status": "n/a",
        "optimizer_status": "n/a",
    }
    astar_metrics: dict[str, Any] = getattr(
        scene, "astar_metrics", default_astar_metrics
    )

    # Pacing: reveal all trees in parallel, finishing together at ~half-time.
    # Fast-record spends the whole duration on racing (no reveal budget).
    half_frames = (
        max(1, max_record_frames // 2) if recording else max(1, fps * 8)
    )
    nodes_per_frame = max(
        1, max(rrt_total, sst_total, astar_total, 1) // half_frames
    )
    if fast_record:
        nodes_per_frame = max(rrt_total, sst_total, astar_total, 1)

    # ---------------------------------------------------------------------------
    # Mutable simulation state
    # ---------------------------------------------------------------------------
    # Discrete background stages for LEFT/RIGHT navigation:
    # 0 = empty init, 1 = all trees complete.
    _bg_stages = [(0, 0, 0), (rrt_total, sst_total, astar_total)]
    _bg_stage_idx = 0

    phase = "background"  # "background" | "racing" | "done"

    rrt_revealed = 0
    sst_revealed = 0
    astar_revealed = 0
    hold = 0
    if fast_record:
        rrt_revealed = rrt_total
        sst_revealed = sst_total
        astar_revealed = astar_total
        _bg_stage_idx = len(_bg_stages) - 1

    rrt_vehicle = None
    sst_vehicle = None
    astar_vehicle = None
    rrt_loop = None
    sst_loop = None
    astar_loop = None
    rrt_traj: list[tuple[float, float, float]] = []
    sst_traj: list[tuple[float, float, float]] = []
    astar_traj: list[tuple[float, float, float]] = []
    rrt_finished = False
    sst_finished = False
    astar_finished = False
    rrt_finish_time: float | None = None
    sst_finish_time: float | None = None
    astar_finish_time: float | None = None
    last_finish_time: float | None = None
    race_time = 0.0

    paused = False

    # CityScene stores simulator config in _sim_cfg; VehicleScene keeps
    # the full YAML under _cfg["simulator"].
    sim_cfg = sim_cfg_early
    tracker_mode = str(sim_cfg.get("tracker", "pure_pursuit"))

    def _start_racing() -> None:
        nonlocal rrt_vehicle, rrt_loop, rrt_traj
        nonlocal sst_vehicle, sst_loop, sst_traj
        nonlocal astar_vehicle, astar_loop, astar_traj
        nonlocal rrt_finished, sst_finished, astar_finished, race_time
        rrt_traj = []
        sst_traj = []
        astar_traj = []
        rrt_finished = False
        sst_finished = False
        astar_finished = False
        race_time = 0.0
        occ = getattr(scene, "_occ", None)
        if tracker_mode == "mpc":
            # One MPC instance per global-planner path.  The planner
            # (RRT* / SST / A*) still owns topology; MPC only tracks the
            # arbitrary waypoint list it receives as reference.
            # City defaults to a longer horizon so anticipation reads on
            # the 600 m video; scenario YAML may still override it.
            mpc_cfg = path_following_mpc_config_from_simulator(
                sim_cfg,
                default_horizon_step_count=DEFAULT_CITY_HORIZON_STEP_COUNT,
                default_horizon_dt=DEFAULT_CITY_HORIZON_DT,
            )
            if rrt_wps:
                rrt_vehicle, rrt_loop = build_vehicle_mpc_sim(
                    rrt_wps, cfg, mpc_cfg, occ
                )
            if sst_wps:
                sst_vehicle, sst_loop = build_vehicle_mpc_sim(
                    sst_wps, cfg, mpc_cfg, occ
                )
            if astar_wps:
                astar_vehicle, astar_loop = build_vehicle_mpc_sim(
                    astar_wps, cfg, mpc_cfg, occ
                )
            logger.info(
                "Race started (tracker=mpc, horizon=%d×%.3fs) for "
                "RRT*/SST/A* references.",
                mpc_cfg.horizon_step_count,
                mpc_cfg.dt,
            )
        else:
            if rrt_wps:
                rrt_vehicle, rrt_loop = build_vehicle_sim(rrt_wps, cfg, occ)
            if sst_wps:
                sst_vehicle, sst_loop = build_vehicle_sim(sst_wps, cfg, occ)
            if astar_wps:
                astar_vehicle, astar_loop = build_vehicle_sim(
                    astar_wps, cfg, occ
                )
            logger.info("Race started (tracker=%s).", tracker_mode)

    def _restart() -> None:
        nonlocal phase, rrt_revealed, sst_revealed, astar_revealed, hold
        nonlocal rrt_finish_time, sst_finish_time, astar_finish_time
        nonlocal last_finish_time, paused
        nonlocal _bg_stage_idx
        phase = "background"
        rrt_revealed = 0
        sst_revealed = 0
        astar_revealed = 0
        hold = 0
        _bg_stage_idx = 0
        rrt_finish_time = None
        sst_finish_time = None
        astar_finish_time = None
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
                            (
                                rrt_revealed,
                                sst_revealed,
                                astar_revealed,
                            ) = _bg_stages[_bg_stage_idx]
                            if (
                                rrt_revealed < rrt_total
                                or sst_revealed < sst_total
                                or astar_revealed < astar_total
                            ):
                                hold = 0
                        elif (
                            event.key == pygame.K_LEFT
                            and phase == "background"
                        ):
                            _bg_stage_idx = max(0, _bg_stage_idx - 1)
                            (
                                rrt_revealed,
                                sst_revealed,
                                astar_revealed,
                            ) = _bg_stages[_bg_stage_idx]
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
                    astar_revealed = min(
                        astar_revealed + nodes_per_frame, astar_total
                    )
                    both_done = (
                        rrt_revealed >= rrt_total
                        and sst_revealed >= sst_total
                        and astar_revealed >= astar_total
                    )
                    if both_done:
                        hold += 1
                        if (
                            fast_record
                            or hold >= _HOLD_FRAMES
                            or (recording and record_frames >= half_frames)
                        ):
                            phase = "racing"
                            _start_racing()
                            logger.info(
                                "Switched to racing phase%s.",
                                " (fast-record)" if fast_record else "",
                            )

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

                    if astar_vehicle is not None and astar_loop is not None:
                        if not astar_finished:
                            astar_loop.step(astar_wps, dt=dt)
                            astar_traj.append(astar_vehicle.pose)
                            gx, gy = astar_wps[-1]
                            if (
                                math.hypot(
                                    astar_vehicle.x - gx,
                                    astar_vehicle.y - gy,
                                )
                                < cfg.goal_radius
                            ):
                                astar_finished = True
                                astar_finish_time = race_time
                                logger.info(
                                    "A* reached goal at t=%.2f s", race_time
                                )

                    all_finished = (
                        rrt_finished
                        and sst_finished
                        and (astar_finished or not astar_wps)
                    )
                    if all_finished and last_finish_time is None:
                        finish_times = [
                            t
                            for t in (
                                rrt_finish_time,
                                sst_finish_time,
                                astar_finish_time if astar_wps else None,
                            )
                            if t is not None
                        ]
                        if finish_times:
                            last_finish_time = max(finish_times)

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

            # World-space GL (content viewport only)
            renderer_gl.setup_2d_projection(
                x_min, x_max, y_min, y_max, layout.content_w, layout.content_h
            )
            layout.setup_content_viewport()

            scene.draw_background(
                rrt_revealed,
                sst_revealed,
                astar_revealed,
                racing=(phase in ("racing", "done")),
            )

            if phase in ("racing", "done"):
                if len(rrt_traj) >= 2:
                    renderer_gl.draw_path(
                        [(p[0], p[1]) for p in rrt_traj],
                        *_c(_C_RRT_TRAJ),
                        width=PAST_TRACE_WIDTH,
                    )
                if len(sst_traj) >= 2:
                    renderer_gl.draw_path(
                        [(p[0], p[1]) for p in sst_traj],
                        *_c(_C_SST_TRAJ),
                        width=PAST_TRACE_WIDTH,
                    )
                if len(astar_traj) >= 2:
                    renderer_gl.draw_path(
                        [(p[0], p[1]) for p in astar_traj],
                        *_c(_C_ASTAR_TRAJ),
                        width=PAST_TRACE_WIDTH,
                    )

                use_mpc = tracker_mode == "mpc"

                def _draw_mpc_prediction(
                    loop: Any,
                    color: tuple[int, int, int],
                    finished: bool,
                ) -> None:
                    if finished or loop is None:
                        return
                    metrics = getattr(loop, "metrics", None)
                    if not isinstance(metrics, dict):
                        return
                    pred = metrics.get("mpc_predicted_xy") or []
                    if len(pred) < 2:
                        return
                    # Brighten the predicted fan so it reads against the
                    # already-drawn optimized route underneath.  No tip disc:
                    # racers stay oriented rectangles only.
                    bright = _brighten_rgb(color, mix=0.70)
                    renderer_gl.draw_path(
                        [(float(p[0]), float(p[1])) for p in pred],
                        *bright,
                        width=PREDICTED_TRACE_WIDTH,
                    )

                if use_mpc:
                    _draw_mpc_prediction(rrt_loop, _C_RRT_VEH, rrt_finished)
                    _draw_mpc_prediction(sst_loop, _C_SST_VEH, sst_finished)
                    _draw_mpc_prediction(
                        astar_loop, _C_ASTAR_VEH, astar_finished
                    )
                else:
                    if rrt_vehicle is not None and not rrt_finished:
                        la = find_lookahead(
                            rrt_vehicle.x,
                            rrt_vehicle.y,
                            rrt_wps,
                            cfg.lookahead_distance,
                        )
                        renderer_gl.draw_disc(
                            la[0], la[1], LOOKAHEAD_DISC_R, *_c(_C_RRT_VEH)
                        )
                    if sst_vehicle is not None and not sst_finished:
                        la = find_lookahead(
                            sst_vehicle.x,
                            sst_vehicle.y,
                            sst_wps,
                            cfg.lookahead_distance,
                        )
                        renderer_gl.draw_disc(
                            la[0], la[1], LOOKAHEAD_DISC_R, *_c(_C_SST_VEH)
                        )
                    if astar_vehicle is not None and not astar_finished:
                        la = find_lookahead(
                            astar_vehicle.x,
                            astar_vehicle.y,
                            astar_wps,
                            cfg.lookahead_distance,
                        )
                        renderer_gl.draw_disc(
                            la[0],
                            la[1],
                            LOOKAHEAD_DISC_R,
                            *_c(_C_ASTAR_VEH),
                        )

                if rrt_vehicle is not None:
                    _draw_race_vehicle(
                        rrt_vehicle.x,
                        rrt_vehicle.y,
                        rrt_vehicle.heading,
                        _C_RRT_VEH,
                    )
                if sst_vehicle is not None:
                    _draw_race_vehicle(
                        sst_vehicle.x,
                        sst_vehicle.y,
                        sst_vehicle.heading,
                        _C_SST_VEH,
                    )
                if astar_vehicle is not None:
                    _draw_race_vehicle(
                        astar_vehicle.x,
                        astar_vehicle.y,
                        astar_vehicle.heading,
                        _C_ASTAR_VEH,
                    )

            layout.reset_viewport()

            # 2-D overlays (full viewport)
            if phase == "background":
                both_ready = (
                    rrt_revealed >= rrt_total
                    and sst_revealed >= sst_total
                    and astar_revealed >= astar_total
                )
                if paused:
                    footer_text = "[ PAUSED ]"
                elif both_ready:
                    footer_text = "All paths ready \u2014 launching race\u2026"
                else:
                    footer_text = "Planning\u2026"
            elif phase == "racing":
                footer_text = (
                    "[ PAUSED ]" if paused else f"Race  {race_time:.1f} s"
                )
            else:  # done
                footer_text = "Press  R  to restart   |   Q  to quit"

            chrome_surf = make_chrome_surface(
                layout, scene.title, footer_text, title_font, font
            )
            renderer_gl.blit_overlay(chrome_surf, 0, 0, sw, sh)

            if phase == "background":
                sidebar_sections = scene.sidebar_content(
                    phase="background",
                    rrt_revealed=rrt_revealed,
                    sst_revealed=sst_revealed,
                    astar_revealed=astar_revealed,
                )
            else:
                sidebar_sections = scene.sidebar_content(
                    phase=phase,
                    race_time=race_time,
                    rrt_finish=rrt_finish_time,
                    sst_finish=sst_finish_time,
                    astar_finish=astar_finish_time,
                )
            draw_sidebar_panel(layout, font, sidebar_sections, sw, sh)

            if phase in ("racing", "done"):
                if rrt_finished or sst_finished or astar_finished:
                    active = [
                        ("RRT*", rrt_finish_time, _C_RRT_HUD),
                        ("SST", sst_finish_time, _C_SST_HUD),
                    ]
                    if astar_wps:
                        active.append(("A*", astar_finish_time, _C_ASTAR_HUD))

                    finished_items = [
                        item for item in active if item[1] is not None
                    ]
                    all_done = len(finished_items) == len(active)
                    if all_done and len(finished_items) >= 2:
                        times = [
                            float(t)
                            for _, t, _ in finished_items
                            if t is not None
                        ]
                        if max(times) - min(times) < 0.15:
                            _draw_winner_banner(
                                big_font,
                                "IT'S A TIE!",
                                _C_TIE,
                                sw,
                                sh,
                                layout=layout,
                            )
                        else:
                            winner = min(
                                finished_items,
                                key=lambda x: (
                                    float(x[1])
                                    if x[1] is not None
                                    else math.inf
                                ),
                            )
                            _draw_winner_banner(
                                big_font,
                                f"{winner[0]}  WINS!",
                                winner[2],
                                sw,
                                sh,
                                layout=layout,
                            )
                    elif finished_items:
                        leader = min(
                            finished_items,
                            key=lambda x: (
                                float(x[1]) if x[1] is not None else math.inf
                            ),
                        )
                        _draw_winner_banner(
                            big_font,
                            f"{leader[0]}  LEADS!",
                            leader[2],
                            sw,
                            sh,
                            layout=layout,
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

    finally:
        if writer is not None:
            writer.close()
        pygame.quit()


def main(cfg: dict, save_path: str | None, sim_duration: float) -> None:
    sim_cfg = load_config("simulator")
    scene_sim_cfg = cfg.get("simulator", {})
    if not isinstance(scene_sim_cfg, dict):
        scene_sim_cfg = {}
    scene = CityScene(
        cfg.get("planner", {}),
        cfg.get("world", {}),
        sim_cfg=scene_sim_cfg,
    )
    run_race(
        scene,
        fps=sim_cfg["fps"],
        dt=sim_cfg["timestep"],
        record=save_path,
        record_duration=sim_duration,
        fast_record=bool(scene_sim_cfg.get("fast_record", False)),
    )
