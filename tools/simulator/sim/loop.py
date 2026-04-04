"""Unified two-phase simulator loop for ARCO planner visualisations.

Phase 1 — **background**: incrementally reveals ``scene.background_total``
items (e.g. exploration-tree nodes).  Skipped when ``background_total`` is 0.

Phase 2 — **tracking**: drives a Dubins vehicle along ``scene.waypoints``.

Entry point: :func:`run_sim`.
"""

from __future__ import annotations

import logging
import math
import os

import pygame
import renderer_gl
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
from sim.camera import CameraFilter, FollowTransform
from sim.scene import SimScene
from sim.tracking import build_vehicle_sim, find_lookahead
from sim.video import VideoWriter

logger = logging.getLogger(__name__)

_DEFAULT_SCREEN_W = 1280
_DEFAULT_SCREEN_H = 800

# Frames to hold the completed background before switching to tracking.
_HOLD_FRAMES = 60

_FOLLOW_ZOOM_DEFAULT = 2.0
_FOLLOW_ZOOM_MIN = 0.4
_FOLLOW_ZOOM_MAX = 8.0
_FOLLOW_ZOOM_STEP = 0.2

# World-space vehicle dimensions for GL rendering.
_VEH_HALF_L = 1.5  # metres
_VEH_HALF_W = 0.7  # metres
_LOOKAHEAD_DISC_R = 0.5  # metres

# Colour constants (float RGB for GL)
_C_TRAJECTORY = (60 / 255, 140 / 255, 220 / 255)
_C_LOOKAHEAD = (240 / 255, 200 / 255, 0 / 255)
_C_VEHICLE = (80 / 255, 220 / 255, 100 / 255)
_C_HUD = (220, 220, 220)
_C_HUD_SHADOW = (40, 40, 50)


def _resolve_screen_size() -> tuple[int, int]:
    """Return a window size derived from the current display, or a fallback.

    Returns:
        ``(width, height)`` in pixels.
    """
    info = pygame.display.Info()
    w = int(getattr(info, "current_w", 0) or 0)
    h = int(getattr(info, "current_h", 0) or 0)
    if w <= 0 or h <= 0:
        logger.warning(
            "Display size unavailable; using fallback %dx%d",
            _DEFAULT_SCREEN_W,
            _DEFAULT_SCREEN_H,
        )
        return (_DEFAULT_SCREEN_W, _DEFAULT_SCREEN_H)
    return (max(640, int(w * 0.9)), max(480, int(h * 0.9)))


def _gl_init_2d(bg_color: tuple[int, int, int]) -> None:
    """Initialise basic OpenGL state for 2-D rendering.

    Args:
        bg_color: Background RGB colour used for ``glClearColor``.
    """
    r, g, b = bg_color[0] / 255.0, bg_color[1] / 255.0, bg_color[2] / 255.0
    glClearColor(r, g, b, 1.0)
    glShadeModel(GL_SMOOTH)
    glDisable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glDisable(GL_LIGHTING)


def _world_bounds(
    tx: object,
    sw: int,
    sh: int,
    follow_zoom: float,
    camera_follow: bool,
    full_bounds: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Compute world-space projection bounds for the current camera mode.

    Args:
        tx: Current camera transform.
        sw: Screen width in pixels.
        sh: Screen height in pixels.
        follow_zoom: Zoom multiplier for follow camera (unused here).
        camera_follow: Whether the follow camera is active.
        full_bounds: Full-scene ``(x_min, x_max, y_min, y_max)`` computed
            once at startup.

    Returns:
        ``(x_min, x_max, y_min, y_max)`` in world metres.
    """
    if not camera_follow:
        return full_bounds
    return renderer_gl.world_bounds_from_transform(tx, sw, sh)


def _draw_tracking_hud(
    font: pygame.font.Font,
    sw: int,
    sh: int,
    label: str,
    step: int,
    speed: float,
    cte: float,
    finished: bool,
    paused: bool,
    extra_lines: list[str] | None = None,
) -> None:
    """Render the vehicle-tracking HUD as a texture overlay.

    Args:
        font: Monospace pygame font.
        sw: Screen width in pixels.
        sh: Screen height in pixels.
        label: Scene label shown at the top.
        step: Current simulation step.
        speed: Current speed in m/s.
        cte: Cross-track error in metres.
        finished: Whether the vehicle has reached the goal.
        paused: Whether the simulation is paused.
        extra_lines: Optional additional text lines appended after the
            standard HUD rows (e.g. a safety-clearance report).
    """
    lines = [
        f"{label} — tracking",
        f"Step: {step}",
        f"Speed: {speed:.1f} m/s",
        f"CTE: {cte:+.1f} m",
    ]
    if paused:
        lines.append("[ PAUSED — press SPACE ]")
    if finished:
        lines.append("[ GOAL REACHED ]")
    if extra_lines:
        lines.extend(extra_lines)

    line_h = font.get_linesize() + 2
    panel_h = len(lines) * line_h + 8
    panel_w = max(font.size(ln)[0] for ln in lines) + 20
    surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    surf.fill((10, 10, 20, 180))
    y = 4
    for line in lines:
        shadow = font.render(line, True, _C_HUD_SHADOW)
        surf.blit(shadow, (11, y + 1))
        text = font.render(line, True, _C_HUD)
        surf.blit(text, (10, y))
        y += line_h
    renderer_gl.blit_overlay(surf, 0, 0, sw, sh)


def run_sim(
    scene: SimScene,
    *,
    fps: int = 30,
    dt: float = 0.1,
    camera: str = "full",
    zoom: bool = False,
    record: str = "",
    record_duration: float = 60.0,
) -> None:
    """Run the unified two-phase simulator loop for *scene*.

    Calls ``scene.build()`` **after** ``pygame.init()`` so scenes may safely
    call ``pygame.font.SysFont`` inside :meth:`~sim.scene.SimScene.build`.

    Keyboard controls (interactive mode only):

    * **SPACE** — pause / resume
    * **R** — restart current phase
    * **C** — toggle camera mode (full / follow-vehicle)
    * **+ / −** — zoom in / out (follow-vehicle mode only)
    * **Q / Escape** — quit

    Args:
        scene: Scene implementing :class:`~sim.scene.SimScene`.
        fps: Target frame rate in frames per second.
        dt: Simulation timestep in seconds per frame.
        camera: Starting camera mode — ``"full"`` for the whole-scene view or
            ``"follow"`` for the vehicle-following zoomed view.
        zoom: If ``True``, fit the initial view to
            ``scene.zoom_world_points`` instead of ``scene.world_points``.
        record: Output MP4 file path for a headless recording.  Empty string
            means interactive mode (opens a window).
        record_duration: Maximum recording length in seconds.
    """
    recording = bool(record)
    max_record_frames = int(fps * record_duration)

    # OpenGL requires a real (or virtual) display — do not set
    # SDL_VIDEODRIVER=dummy.  For headless recording use xvfb-run.
    if recording:
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    pygame.init()

    if recording:
        screen_w, screen_h = _DEFAULT_SCREEN_W, _DEFAULT_SCREEN_H
    else:
        screen_w, screen_h = _resolve_screen_size()
        logger.info("Window size: %dx%d", screen_w, screen_h)

    screen_size = (screen_w, screen_h)
    pygame.display.set_mode(screen_size, pygame.OPENGL | pygame.DOUBLEBUF)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)

    # Build scene AFTER pygame.init() so SysFont is safe to call.
    scene.build()
    pygame.display.set_caption(scene.title)

    # Initialise GL state using scene background colour.
    _gl_init_2d(scene.bg_color)

    # Compute full-scene world bounds from scene world points.
    ref_pts = scene.zoom_world_points if zoom else scene.world_points
    _all_x = [p[0] for p in ref_pts]
    _all_y = [p[1] for p in ref_pts]
    full_bounds = (
        min(_all_x),
        max(_all_x),
        min(_all_y),
        max(_all_y),
    )

    # Background reveal pacing.
    background_total = scene.background_total
    half_frames = max(1, max_record_frames // 2)
    nodes_per_frame = (
        max(1, background_total // half_frames) if background_total > 0 else 0
    )

    # Camera filter starts at the first waypoint (or world origin).
    waypoints = scene.waypoints
    cam_start_x, cam_start_y = waypoints[0] if waypoints else (0.0, 0.0)
    cam_filter = CameraFilter(cam_start_x, cam_start_y)
    camera_follow = camera == "follow"
    follow_zoom = _FOLLOW_ZOOM_DEFAULT

    # Compute a scale for the full view (pixels per metre) for follow camera.
    world_w = max(full_bounds[1] - full_bounds[0], 1.0)
    world_h = max(full_bounds[3] - full_bounds[2], 1.0)
    full_scale = min(
        (screen_w - 120) / world_w,
        (screen_h - 120) / world_h,
    )

    def _current_follow_bounds() -> tuple[float, float, float, float]:
        scale = full_scale * follow_zoom
        half_w_world = screen_w / (2.0 * scale)
        half_h_world = screen_h / (2.0 * scale)
        cx, cy = cam_filter.x, cam_filter.y
        return (
            cx - half_w_world,
            cx + half_w_world,
            cy - half_h_world,
            cy + half_h_world,
        )

    def _setup_projection() -> None:
        if camera_follow:
            x_min, x_max, y_min, y_max = _current_follow_bounds()
        else:
            x_min, x_max, y_min, y_max = full_bounds
        renderer_gl.setup_2d_projection(
            x_min, x_max, y_min, y_max, screen_w, screen_h
        )

    # Phase and vehicle state.
    phase = "background" if background_total > 0 else "tracking"
    revealed = 0
    hold = 0

    vehicle = None
    veh_loop = None
    trajectory: list[tuple[float, float, float]] = []
    veh_step = 0
    veh_finished = False
    paused = False

    def _start_tracking() -> None:
        nonlocal vehicle, veh_loop, trajectory, veh_step, veh_finished
        if not waypoints:
            veh_finished = True
            logger.warning("Scene has no waypoints; skipping tracking.")
            return
        cfg = scene.vehicle_config
        v, lp = build_vehicle_sim(waypoints, cfg)
        vehicle = v
        veh_loop = lp
        trajectory = []
        veh_step = 0
        veh_finished = False
        if camera_follow:
            cam_filter.reset(waypoints[0][0], waypoints[0][1])

    def _restart() -> None:
        nonlocal phase, revealed, hold, paused
        if background_total > 0:
            phase = "background"
            revealed = 0
            hold = 0
        else:
            phase = "tracking"
            _start_tracking()
        paused = False

    # Initialise immediately if no background phase.
    if phase == "tracking":
        _start_tracking()

    writer: VideoWriter | None = (
        VideoWriter(record, screen_w, screen_h, fps) if recording else None
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
                        elif event.key == pygame.K_c:
                            camera_follow = not camera_follow
                            logger.info(
                                "Camera mode: %s",
                                "follow" if camera_follow else "full",
                            )
                            if camera_follow and waypoints:
                                cam_filter.reset(
                                    waypoints[0][0], waypoints[0][1]
                                )
                        elif event.key in (
                            pygame.K_PLUS,
                            pygame.K_EQUALS,
                            pygame.K_KP_PLUS,
                        ):
                            follow_zoom = min(
                                _FOLLOW_ZOOM_MAX,
                                follow_zoom + _FOLLOW_ZOOM_STEP,
                            )
                        elif event.key in (
                            pygame.K_MINUS,
                            pygame.K_KP_MINUS,
                        ):
                            follow_zoom = max(
                                _FOLLOW_ZOOM_MIN,
                                follow_zoom - _FOLLOW_ZOOM_STEP,
                            )

            # ------------------------------------------------------------------
            # Phase logic
            # ------------------------------------------------------------------
            if not paused:
                if phase == "background":
                    if revealed < background_total:
                        revealed = min(
                            revealed + nodes_per_frame, background_total
                        )
                    else:
                        hold += 1
                        transition = hold >= _HOLD_FRAMES or (
                            recording and record_frames >= half_frames
                        )
                        if transition:
                            phase = "tracking"
                            _start_tracking()
                            logger.info("Switched to tracking phase.")
                elif phase == "tracking" and not veh_finished:
                    if vehicle is not None and veh_loop is not None:
                        veh_loop.step(waypoints, dt=dt)
                        veh_step += 1
                        trajectory.append(vehicle.pose)
                        gx, gy = waypoints[-1]
                        cfg = scene.vehicle_config
                        if (
                            math.hypot(vehicle.x - gx, vehicle.y - gy)
                            < cfg.goal_radius
                        ):
                            veh_finished = True
                            logger.info("Goal reached in %d steps.", veh_step)

            # Camera filter update.
            if camera_follow and vehicle is not None:
                cam_filter.update(float(vehicle.x), float(vehicle.y), dt)

            # ------------------------------------------------------------------
            # Render
            # ------------------------------------------------------------------
            r_bg, g_bg, b_bg = scene.bg_color
            glClearColor(r_bg / 255.0, g_bg / 255.0, b_bg / 255.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT)

            _setup_projection()

            reveal_count = (
                revealed if phase == "background" else background_total
            )
            scene.draw_background(reveal_count)

            if phase == "background":
                scene.draw_background_hud(font, screen_w, screen_h, revealed)
            elif vehicle is not None:
                if len(trajectory) >= 2:
                    renderer_gl.draw_path(
                        [(p[0], p[1]) for p in trajectory],
                        *_C_TRAJECTORY,
                        width=1.5,
                    )
                la = find_lookahead(
                    vehicle.x,
                    vehicle.y,
                    waypoints,
                    scene.vehicle_config.lookahead_distance,
                )
                renderer_gl.draw_disc(
                    la[0], la[1], _LOOKAHEAD_DISC_R, *_C_LOOKAHEAD
                )
                renderer_gl.draw_oriented_rect(
                    vehicle.x,
                    vehicle.y,
                    _VEH_HALF_L,
                    _VEH_HALF_W,
                    vehicle.heading,
                    *_C_VEHICLE,
                )
                metrics = (
                    (veh_loop.metrics or {}) if veh_loop is not None else {}
                )
                _draw_tracking_hud(
                    font,
                    screen_w,
                    screen_h,
                    scene.title,
                    veh_step,
                    float(metrics.get("speed", 0.0)),
                    float(metrics.get("cross_track_error", 0.0)),
                    veh_finished,
                    paused,
                    extra_lines=(
                        scene.finish_hud_lines if veh_finished else None
                    ),
                )

            # ------------------------------------------------------------------
            # Output: record frame or flip display
            # ------------------------------------------------------------------
            if recording and writer is not None:
                pygame.display.flip()
                writer.write_frame_gl()
                record_frames += 1
                if record_frames >= max_record_frames:
                    running = False
                elif phase == "tracking" and veh_finished:
                    running = False
            else:
                pygame.display.flip()
                clock.tick(fps)
    finally:
        if writer is not None:
            writer.close()
        pygame.quit()
