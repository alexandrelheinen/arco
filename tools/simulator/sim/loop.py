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
from renderer import (
    WorldTransform,
    draw_tracking_hud,
    draw_tracking_target,
    draw_trajectory,
    draw_vehicle,
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

    if recording:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    pygame.init()

    if recording:
        screen_w, screen_h = _DEFAULT_SCREEN_W, _DEFAULT_SCREEN_H
    else:
        screen_w, screen_h = _resolve_screen_size()
        logger.info("Window size: %dx%d", screen_w, screen_h)

    screen_size = (screen_w, screen_h)
    screen = pygame.display.set_mode(screen_size)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)

    # Build scene AFTER pygame.init() so SysFont is safe to call.
    scene.build()
    pygame.display.set_caption(scene.title)

    # Transforms: full view and optional zoom view.
    ref_pts = scene.zoom_world_points if zoom else scene.world_points
    full_transform = WorldTransform(ref_pts, screen_size, margin=60)

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

    def _current_transform() -> WorldTransform | FollowTransform:
        if not camera_follow:
            return full_transform
        return FollowTransform(
            cam_filter.x,
            cam_filter.y,
            screen_size,
            full_transform.scale * follow_zoom,
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
            screen.fill(scene.bg_color)
            tx = _current_transform()

            reveal_count = (
                revealed if phase == "background" else background_total
            )
            scene.draw_background(screen, tx, reveal_count)

            if phase == "background":
                scene.draw_background_hud(screen, font, revealed)
            elif vehicle is not None:
                if len(trajectory) >= 2:
                    draw_trajectory(screen, trajectory, tx)
                la = find_lookahead(
                    vehicle.x,
                    vehicle.y,
                    waypoints,
                    scene.vehicle_config.lookahead_distance,
                )
                draw_tracking_target(screen, la, tx)
                draw_vehicle(screen, vehicle.x, vehicle.y, vehicle.heading, tx)
                metrics = (
                    (veh_loop.metrics or {}) if veh_loop is not None else {}
                )
                draw_tracking_hud(
                    screen,
                    font,
                    scene.title,
                    veh_step,
                    float(metrics.get("speed", 0.0)),
                    float(metrics.get("cross_track_error", 0.0)),
                    veh_finished,
                    paused,
                )

            # ------------------------------------------------------------------
            # Output: record frame or flip display
            # ------------------------------------------------------------------
            if recording and writer is not None:
                writer.write_frame(screen)
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
