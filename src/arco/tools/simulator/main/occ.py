#!/usr/bin/env python
"""Object-centric control — piano movers 2-D simulator.

A 2-D rigid body (square or circle) is transported from pose A to pose B
by N mobile actuators. The path is pre-planned in the (x, y, ψ) C-space
by RRT* (left panel) and SST (right panel) and then tracked by an
ActuatorArray PD controller.

Controls
--------
SPACE    Pause / resume
R        Restart
Q/Esc    Quit

Usage
-----
::

    cd tools/simulator
    python main/occ.py

Record::

    xvfb-run -a python main/occ.py --record /tmp/occ.mp4 --record-duration 30
"""

from __future__ import annotations

import argparse
import logging
import math
import sys

import numpy as np
import pygame

from arco.control import ActuatorArray
from arco.control.rigid_body import CircleBody, SquareBody
from arco.mapping import KDTreeOccupancy
from arco.tools.config import load_config
from arco.tools.simulator.logging_config import configure_logging
from arco.tools.simulator.scenes.occ import OCCScene

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def _world_to_screen(
    pos: tuple[float, float],
    origin: tuple[float, float],
    scale: float,
) -> tuple[int, int]:
    """Convert world coordinates to screen pixel coordinates.

    Args:
        pos: World position ``(x, y)``.
        origin: Screen pixel position of world origin ``(ox, oy)``.
        scale: Pixels per metre.

    Returns:
        Screen pixel ``(px, py)`` with y-axis flipped.
    """
    ox, oy = origin
    wx, wy = pos
    return (int(ox + wx * scale), int(oy - wy * scale))


def _draw_body(
    surface: pygame.Surface,
    body: object,
    origin: tuple[float, float],
    scale: float,
    color: tuple[int, int, int],
) -> None:
    """Draw a rigid body (square or circle) on the surface.

    Args:
        surface: Pygame surface to draw on.
        body: A ``SquareBody`` or ``CircleBody`` instance.
        origin: Screen origin in pixels.
        scale: Pixels per metre.
        color: RGB fill color.
    """
    if isinstance(body, SquareBody):
        corners = body.corners()
        pts = [_world_to_screen((c[0], c[1]), origin, scale) for c in corners]
        pygame.draw.polygon(surface, color, pts)
        pygame.draw.polygon(surface, (0, 0, 0), pts, 2)
    elif isinstance(body, CircleBody):
        cx, cy = float(body.pose[0]), float(body.pose[1])
        sc = _world_to_screen((cx, cy), origin, scale)
        r_px = max(2, int(body.radius * scale))
        pygame.draw.circle(surface, color, sc, r_px)
        pygame.draw.circle(surface, (0, 0, 0), sc, r_px, 2)


def _draw_obstacle(
    surface: pygame.Surface,
    obs: list[float],
    origin: tuple[float, float],
    scale: float,
    color: tuple[int, int, int] = (180, 60, 60),
) -> None:
    """Draw an AABB obstacle as a filled rectangle.

    Args:
        surface: Pygame surface to draw on.
        obs: ``[x_min, y_min, x_max, y_max]``.
        origin: Screen origin in pixels.
        scale: Pixels per metre.
        color: RGB fill color.
    """
    xmin, ymin, xmax, ymax = obs
    tl = _world_to_screen((xmin, ymax), origin, scale)
    br = _world_to_screen((xmax, ymin), origin, scale)
    rect = pygame.Rect(tl[0], tl[1], br[0] - tl[0], br[1] - tl[1])
    pygame.draw.rect(surface, color, rect)
    pygame.draw.rect(surface, (0, 0, 0), rect, 2)


def _draw_path(
    surface: pygame.Surface,
    path: list[np.ndarray] | None,
    origin: tuple[float, float],
    scale: float,
    color: tuple[int, int, int],
    width: int = 2,
) -> None:
    """Draw a planned path as a polyline.

    Args:
        surface: Pygame surface to draw on.
        path: Sequence of ``[x, y, psi]`` waypoints, or ``None``.
        origin: Screen origin in pixels.
        scale: Pixels per metre.
        color: RGB line color.
        width: Line width in pixels.
    """
    if not path or len(path) < 2:
        return
    pts = [
        _world_to_screen((float(p[0]), float(p[1])), origin, scale)
        for p in path
    ]
    pygame.draw.lines(surface, color, False, pts, width)


def _draw_actuators(
    surface: pygame.Surface,
    positions: np.ndarray,
    origin: tuple[float, float],
    scale: float,
    color: tuple[int, int, int] = (80, 200, 120),
) -> None:
    """Draw actuator positions as small circles.

    Args:
        surface: Pygame surface to draw on.
        positions: Array of shape (N, 2) with world positions.
        origin: Screen origin in pixels.
        scale: Pixels per metre.
        color: RGB fill color.
    """
    for pos in positions:
        sc = _world_to_screen((float(pos[0]), float(pos[1])), origin, scale)
        pygame.draw.circle(surface, color, sc, 5)
        pygame.draw.circle(surface, (0, 0, 0), sc, 5, 1)


def _draw_pose_marker(
    surface: pygame.Surface,
    pose: np.ndarray,
    origin: tuple[float, float],
    scale: float,
    color: tuple[int, int, int],
    label: str = "",
) -> None:
    """Draw a pose marker (cross + heading arrow).

    Args:
        surface: Pygame surface.
        pose: ``[x, y, psi]``.
        origin: Screen origin.
        scale: Pixels per metre.
        color: RGB colour.
        label: Optional text label.
    """
    sc = _world_to_screen((float(pose[0]), float(pose[1])), origin, scale)
    pygame.draw.circle(surface, color, sc, 8, 2)
    # Heading arrow
    length = 20
    ex = sc[0] + int(length * math.cos(float(pose[2])))
    ey = sc[1] - int(length * math.sin(float(pose[2])))
    pygame.draw.line(surface, color, sc, (ex, ey), 2)


# ---------------------------------------------------------------------------
# Controller helper
# ---------------------------------------------------------------------------


def _wrap_angle(angle: float) -> float:
    """Wrap angle to [-π, π].

    Args:
        angle: Angle in radians.

    Returns:
        Wrapped angle in [-π, π].
    """
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _compute_desired_wrench(
    body: object,
    goal: np.ndarray,
    ctrl_cfg: dict,
) -> np.ndarray:
    """Compute desired body wrench from PD control law.

    Args:
        body: The rigid body being controlled.
        goal: Target pose ``[x, y, psi]``.
        ctrl_cfg: Control config with kp_pos, kd_pos, kp_psi, kd_psi.

    Returns:
        Desired wrench ``[Fx, Fy, torque]``.
    """
    from arco.control.rigid_body.base import RigidBody

    assert isinstance(body, RigidBody)
    pose = body.pose
    vel = body.velocity
    kp_pos = float(ctrl_cfg.get("kp_pos", 2.5))
    kd_pos = float(ctrl_cfg.get("kd_pos", 4.2))
    kp_psi = float(ctrl_cfg.get("kp_psi", 0.50))
    kd_psi = float(ctrl_cfg.get("kd_psi", 0.55))
    ex = float(goal[0]) - float(pose[0])
    ey = float(goal[1]) - float(pose[1])
    e_psi = _wrap_angle(float(goal[2]) - float(pose[2]))
    fx = kp_pos * ex - kd_pos * float(vel[0])
    fy = kp_pos * ey - kd_pos * float(vel[1])
    tau = kp_psi * e_psi - kd_psi * float(vel[2])
    return np.array([fx, fy, tau])


# ---------------------------------------------------------------------------
# Cartesian obstacle occupancy
# ---------------------------------------------------------------------------


def _build_cartesian_occupancy(
    obstacles: list[list[float]],
    n_samples: int = 32,
) -> KDTreeOccupancy:
    """Build a 2-D KDTreeOccupancy by sampling AABB obstacle boundaries.

    Args:
        obstacles: List of ``[x_min, y_min, x_max, y_max]`` boxes.
        n_samples: Number of sample points per side per obstacle.

    Returns:
        :class:`~arco.mapping.KDTreeOccupancy` suitable for
        ``nearest_obstacle`` queries.
    """
    pts: list[list[float]] = []
    for obs in obstacles:
        xmin, ymin, xmax, ymax = obs
        dx = xmax - xmin
        dy = ymax - ymin
        ts = np.linspace(0.0, 1.0, n_samples, endpoint=False)
        for t in ts:
            pts.append([xmin + t * dx, ymin])
            pts.append([xmin + t * dx, ymax])
            pts.append([xmin, ymin + t * dy])
            pts.append([xmax, ymin + t * dy])
    if not pts:
        pts = [[1e9, 1e9]]
    return KDTreeOccupancy(np.array(pts, dtype=float), clearance=1e-3)


# ---------------------------------------------------------------------------
# Loading screen
# ---------------------------------------------------------------------------


def _show_loading(
    screen: pygame.Surface,
    font: pygame.font.Font,
    msg: str,
    step: int,
    total: int,
) -> None:
    """Draw a simple loading progress screen.

    Args:
        screen: Main pygame surface.
        font: Font for text rendering.
        msg: Status message string.
        step: Current step index.
        total: Total number of steps.
    """
    w, h = screen.get_size()
    screen.fill((30, 30, 30))
    text = font.render(
        f"Loading ({step}/{total}): {msg}", True, (220, 220, 220)
    )
    screen.blit(text, ((w - text.get_width()) // 2, h // 2))
    bar_w = int((w - 80) * step / max(total, 1))
    pygame.draw.rect(screen, (80, 80, 200), (40, h // 2 + 40, bar_w, 12))
    pygame.draw.rect(screen, (120, 120, 120), (40, h // 2 + 40, w - 80, 12), 2)
    pygame.display.flip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    p = argparse.ArgumentParser(description="OCC 2-D piano movers simulator")
    p.add_argument("--record", metavar="PATH", help="Save video to PATH")
    p.add_argument(
        "--record-duration",
        type=float,
        default=30.0,
        metavar="SECS",
        help="Recording duration in seconds (default: 30)",
    )
    p.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Window width (default: 1280)",
    )
    p.add_argument(
        "--height",
        type=int,
        default=640,
        help="Window height (default: 640)",
    )
    return p.parse_args()


def main() -> None:
    """Run the OCC 2-D piano movers simulator."""
    configure_logging()
    args = _parse_args()

    cfg = load_config("occ")
    env_cfg: dict = cfg.get("environment", {})
    ctrl_cfg: dict = cfg.get("control", {})
    sim_cfg: dict = cfg.get("simulator", {})
    dt = float(sim_cfg.get("dt", 0.02))

    pygame.init()
    flags = pygame.DOUBLEBUF
    screen = pygame.display.set_mode((args.width, args.height), flags)
    pygame.display.set_caption("OCC — Piano Movers")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)
    title_font = pygame.font.SysFont("monospace", 18, bold=True)

    # Build scene with loading screen feedback
    scene = OCCScene(cfg)

    def _progress(msg: str, step: int, total: int) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
        _show_loading(screen, font, msg, step, total)

    scene.build(progress=_progress)

    body = scene.body
    actuators = scene.actuators
    obstacles = scene.obstacles

    occ_2d = _build_cartesian_occupancy(obstacles)
    k_rep = float(ctrl_cfg.get("k_rep", 5.0))
    d0 = 4.0 * float(cfg.get("actuator", {}).get("standoff", 0.05))

    x_range = [float(v) for v in env_cfg.get("x_range", [-4, 4])]
    y_range = [float(v) for v in env_cfg.get("y_range", [-3, 3])]

    # Compute scale and origin for one panel (we split width in half)
    panel_w = args.width // 2
    panel_h = args.height
    x_span = x_range[1] - x_range[0]
    y_span = y_range[1] - y_range[0]
    scale = min(
        (panel_w - 40) / max(x_span, 1e-3),
        (panel_h - 60) / max(y_span, 1e-3),
    )
    cx_world = (x_range[0] + x_range[1]) / 2.0
    cy_world = (y_range[0] + y_range[1]) / 2.0
    left_origin = (
        panel_w // 2 - int(cx_world * scale),
        panel_h // 2 + int(cy_world * scale),
    )
    right_origin = (
        panel_w + panel_w // 2 - int(cx_world * scale),
        panel_h // 2 + int(cy_world * scale),
    )

    # Path tracking state
    start_pose = scene.start_pose.copy()
    goal_pose = scene.goal_pose.copy()

    def _make_waypoints(
        path: list[np.ndarray] | None,
    ) -> list[np.ndarray]:
        if path and len(path) >= 2:
            return list(path)
        return [start_pose, goal_pose]

    rrt_waypoints = _make_waypoints(
        scene.rrt_traj if scene.rrt_traj else scene.rrt_path
    )
    sst_waypoints = _make_waypoints(
        scene.sst_traj if scene.sst_traj else scene.sst_path
    )

    def _reset_bodies() -> tuple[object, object, int, int]:
        body_type = str(cfg.get("body", {}).get("type", "square"))
        mass = float(cfg.get("body", {}).get("mass", 5.0))
        act_cfg = cfg.get("actuator", {})
        count = int(act_cfg.get("count", 3))
        standoff = float(act_cfg.get("standoff", 0.05))
        omega = float(act_cfg.get("omega", 10.0))
        zeta = float(act_cfg.get("zeta", 0.7))
        spring_stiffness = float(act_cfg.get("spring_stiffness", 100.0))

        if body_type == "square":
            side = float(cfg.get("body", {}).get("side_length", 0.5))
            b = SquareBody(
                mass=mass,
                side_length=side,
                x=float(start_pose[0]),
                y=float(start_pose[1]),
                psi=float(start_pose[2]),
            )
        else:
            radius = float(cfg.get("body", {}).get("radius", 0.3))
            b = CircleBody(
                mass=mass,
                radius=radius,
                x=float(start_pose[0]),
                y=float(start_pose[1]),
                psi=float(start_pose[2]),
            )
        a = ActuatorArray(
            actuator_count=count,
            standoff=standoff,
            omega=omega,
            zeta=zeta,
            spring_stiffness=spring_stiffness,
        )
        a.init_radii(b)
        return b, a, 0, 0

    rrt_body, rrt_acts, rrt_wp_idx, _ = _reset_bodies()
    sst_body, sst_acts, sst_wp_idx, _ = _reset_bodies()

    paused = False
    recording = args.record is not None
    video_writer = None
    if recording:
        try:
            from sim.video import VideoWriter

            video_writer = VideoWriter(
                args.record,
                args.width,
                args.height,
                int(1.0 / dt),
            )
            video_writer.open()
        except Exception as exc:
            logger.warning("Cannot record video: %s", exc)
            recording = False

    record_frames = int(args.record_duration / dt)
    frame_count = 0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    rrt_body, rrt_acts, rrt_wp_idx, _ = _reset_bodies()
                    sst_body, sst_acts, sst_wp_idx, _ = _reset_bodies()
                    rrt_body.reset(
                        x=float(start_pose[0]),
                        y=float(start_pose[1]),
                        psi=float(start_pose[2]),
                    )
                    sst_body.reset(
                        x=float(start_pose[0]),
                        y=float(start_pose[1]),
                        psi=float(start_pose[2]),
                    )

        if not paused:
            goal_tol = float(
                cfg.get("planner", {}).get("goal_tolerance", 0.25)
            )

            # Advance RRT* body
            if rrt_wp_idx < len(rrt_waypoints):
                rrt_goal = rrt_waypoints[
                    min(rrt_wp_idx, len(rrt_waypoints) - 1)
                ]
                w = _compute_desired_wrench(rrt_body, rrt_goal, ctrl_cfg)
                sst_peers = np.vstack(
                    [
                        sst_acts.actuator_positions(sst_body),
                        sst_body.pose[:2].reshape(1, 2),
                    ]
                )
                w = w + rrt_acts.repulsive_wrench(
                    rrt_body,
                    occ_2d.nearest_obstacle,
                    sst_peers,
                    k_rep=k_rep,
                    d0=d0,
                )
                # Step 3: reference angles from gradient descent
                rrt_acts.update_angles_for_target(rrt_body, w)
                # Step 4: radial-only allocation at ACTUAL angles → spring inversion
                f_des = rrt_acts.allocate_radial_forces(w, rrt_body)
                rrt_acts.compute_ref_radii(rrt_body, f_des)
                # Integrate actuator second-order dynamics
                rrt_acts.step_actuators(dt)
                # Apply spring forces to body
                rrt_acts.apply_spring_forces_to_body(rrt_body)
                rrt_body.step(dt)
                pose = rrt_body.pose
                dist = math.hypot(
                    float(pose[0]) - float(rrt_goal[0]),
                    float(pose[1]) - float(rrt_goal[1]),
                )
                if dist < goal_tol:
                    rrt_wp_idx = min(rrt_wp_idx + 1, len(rrt_waypoints))

            # Advance SST body
            if sst_wp_idx < len(sst_waypoints):
                sst_goal = sst_waypoints[
                    min(sst_wp_idx, len(sst_waypoints) - 1)
                ]
                w = _compute_desired_wrench(sst_body, sst_goal, ctrl_cfg)
                rrt_peers = np.vstack(
                    [
                        rrt_acts.actuator_positions(rrt_body),
                        rrt_body.pose[:2].reshape(1, 2),
                    ]
                )
                w = w + sst_acts.repulsive_wrench(
                    sst_body,
                    occ_2d.nearest_obstacle,
                    rrt_peers,
                    k_rep=k_rep,
                    d0=d0,
                )
                # Step 3: reference angles from gradient descent
                sst_acts.update_angles_for_target(sst_body, w)
                # Step 4: radial-only allocation at ACTUAL angles → spring inversion
                f_des = sst_acts.allocate_radial_forces(w, sst_body)
                sst_acts.compute_ref_radii(sst_body, f_des)
                # Integrate actuator second-order dynamics
                sst_acts.step_actuators(dt)
                # Apply spring forces to body
                sst_acts.apply_spring_forces_to_body(sst_body)
                sst_body.step(dt)
                pose = sst_body.pose
                dist = math.hypot(
                    float(pose[0]) - float(sst_goal[0]),
                    float(pose[1]) - float(sst_goal[1]),
                )
                if dist < goal_tol:
                    sst_wp_idx = min(sst_wp_idx + 1, len(sst_waypoints))

        # Draw
        screen.fill((40, 40, 40))

        # Divider
        pygame.draw.line(
            screen,
            (120, 120, 120),
            (panel_w, 0),
            (panel_w, panel_h),
            2,
        )

        for panel_idx, (origin, body_s, acts_s, path_s, label) in enumerate(
            [
                (
                    left_origin,
                    rrt_body,
                    rrt_acts,
                    scene.rrt_path,
                    "RRT*",
                ),
                (
                    right_origin,
                    sst_body,
                    sst_acts,
                    scene.sst_path,
                    "SST",
                ),
            ]
        ):
            # Obstacles
            for obs in obstacles:
                _draw_obstacle(screen, obs, origin, scale)

            # Planned path
            _draw_path(screen, path_s, origin, scale, (100, 180, 255), width=1)

            # Start/goal markers
            _draw_pose_marker(
                screen, start_pose, origin, scale, (0, 200, 80), "S"
            )
            _draw_pose_marker(
                screen, goal_pose, origin, scale, (200, 200, 0), "G"
            )

            # Actuators
            positions = acts_s.actuator_positions(body_s)
            _draw_actuators(screen, positions, origin, scale)

            # Body
            _draw_body(screen, body_s, origin, scale, (60, 120, 220))

            # Label
            lbl = title_font.render(label, True, (240, 240, 240))
            screen.blit(lbl, (panel_idx * panel_w + 10, 10))

        # Status bar
        status = "PAUSED" if paused else "RUNNING"
        info = font.render(
            f"{status} | SPACE=pause  R=restart  Q=quit",
            True,
            (180, 180, 180),
        )
        screen.blit(info, (10, panel_h - 24))

        pygame.display.flip()

        if recording and video_writer is not None:
            video_writer.write_frame(screen)
            frame_count += 1
            if frame_count >= record_frames:
                running = False

        clock.tick(int(1.0 / dt))

    if video_writer is not None:
        try:
            video_writer.close()
        except Exception:
            pass
    pygame.quit()


if __name__ == "__main__":
    main()
