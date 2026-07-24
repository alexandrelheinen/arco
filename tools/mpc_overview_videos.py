#!/usr/bin/env python3
"""Generate three overview MP4s showing MPC tracking in practice.

Scenarios (representative of the ARCO simulator stack, headless matplotlib):

1. ``mpc_overview_dubins.mp4`` — SE(2) Dubins path-following MPC through a
   warehouse pinch (SE(2) race-tracking demo, same role as city NMPCC).
2. ``mpc_overview_ppp.mp4`` — 3-DOF Cartesian gantry (PPP) with
   :class:`JointSpaceMPC` through barrier boxes.
3. ``mpc_overview_rrp.mp4`` — RRP / SCARA-like arm with
   :class:`JointSpaceMPC` navigating pillars in joint space (FK overlay).

Usage::

    python tools/mpc_overview_videos.py
    python tools/mpc_overview_videos.py --out-dir tools/output --fps 20
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation

from arco.control.mpc import (
    DubinsPathFollowingMPC,
    DubinsVehicleLimits,
    JointSpaceMPC,
    JointSpaceMPCConfig,
    MPCTrackingLoop,
    PathFollowingMPCConfig,
)
from arco.guidance.vehicle import DubinsVehicle
from arco.mapping.kdtree import KDTreeOccupancy


def _write_animation(
    fig: plt.Figure,
    update,
    frame_count: int,
    out_path: Path,
    fps: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, bitrate=1800)
    anim = FuncAnimation(fig, update, frames=frame_count, interval=1000 / fps)
    anim.save(str(out_path), writer=writer)
    plt.close(fig)
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# 1) Dubins pinch (vehicle race analogue)
# ---------------------------------------------------------------------------


def _pinch_occupancy(clearance: float = 0.30) -> KDTreeOccupancy:
    pts: list[list[float]] = []
    for x in np.linspace(4.0, 5.5, 16):
        for y in np.linspace(-1.2, -0.20, 12):
            pts.append([float(x), float(y)])
        for y in np.linspace(0.20, 1.2, 12):
            pts.append([float(x), float(y)])
    return KDTreeOccupancy(np.array(pts, dtype=float), clearance=clearance)


def generate_dubins_video(out_path: Path, fps: int = 20) -> None:
    """Top-down Dubins MPC through a pinch corridor."""
    cruise = 0.36
    dt = 0.05
    path = [(float(x), 0.0) for x in np.arange(0.0, 12.0, 0.25)]
    occ = _pinch_occupancy(0.30)
    vehicle = DubinsVehicle(
        x=0.0,
        y=0.0,
        heading=0.0,
        max_speed=cruise + 0.2,
        min_speed=0.05,
        max_turn_rate=1.2,
        max_acceleration=1.2,
        max_turn_rate_dot=2.0,
    )
    vehicle._speed = cruise
    cfg = PathFollowingMPCConfig(
        horizon_step_count=20,
        dt=dt,
        cruise_speed=cruise,
        weight_obstacle=120.0,
        weight_contour=8.0,
        weight_progress=1.0,
        max_solver_iter_count=60,
    )
    tracker = DubinsPathFollowingMPC(
        vehicle_limits=DubinsVehicleLimits(
            max_speed=cruise + 0.2,
            min_speed=0.05,
            max_turn_rate=1.2,
            max_acceleration=1.2,
            max_turn_rate_dot=2.0,
        ),
        config=cfg,
        occupancy=occ,
    )
    loop = MPCTrackingLoop(vehicle, tracker, cruise_speed=cruise)

    steps = int(14.0 / dt)
    poses = np.zeros((steps, 3))
    speeds = np.zeros(steps)
    clearances = np.zeros(steps)
    for i in range(steps):
        m = loop.step(path, dt=dt)
        poses[i] = m["pose"]
        speeds[i] = m["speed"]
        dist, _ = occ.nearest_obstacle(np.array(m["pose"][:2]))
        clearances[i] = dist

    fig, (ax, ax_s) = plt.subplots(
        2,
        1,
        figsize=(8.0, 6.2),
        gridspec_kw={"height_ratios": [3.0, 1.2]},
    )
    obs_xy = occ.points
    ax.scatter(obs_xy[:, 0], obs_xy[:, 1], s=8, c="#555555", label="obstacles")
    ax.plot(
        [p[0] for p in path],
        [p[1] for p in path],
        "--",
        color="#888888",
        lw=1.0,
        label="reference",
    )
    (trail,) = ax.plot([], [], color="#c45c26", lw=2.0, label="MPC trail")
    (veh,) = ax.plot([], [], "o", color="#c45c26", ms=8)
    heading_line = ax.plot([], [], color="#1f4e79", lw=2.0)[0]
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.set_title("Dubins path-following MPC (vehicle / pinch corridor)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    (speed_line,) = ax_s.plot([], [], color="#c45c26", label="speed")
    ax_s.axhline(cruise, color="gray", ls="--", lw=0.8)
    ax_s.set_xlim(0, steps * dt)
    ax_s.set_ylim(0, cruise * 1.2)
    ax_s.set_xlabel("time (s)")
    ax_s.set_ylabel("speed (m/s)")
    ax_s.grid(True, alpha=0.3)
    fig.tight_layout()

    def update(frame: int):
        i = min(frame, steps - 1)
        trail.set_data(poses[: i + 1, 0], poses[: i + 1, 1])
        x, y, th = poses[i]
        veh.set_data([x], [y])
        heading_line.set_data(
            [x, x + 0.4 * math.cos(th)], [y, y + 0.4 * math.sin(th)]
        )
        t = np.arange(i + 1) * dt
        speed_line.set_data(t, speeds[: i + 1])
        return trail, veh, heading_line, speed_line

    _write_animation(fig, update, steps, out_path, fps)


# ---------------------------------------------------------------------------
# 2) PPP gantry (Cartesian 3-DOF)
# ---------------------------------------------------------------------------


def _ppp_boxes() -> list[tuple[float, float, float, float, float, float]]:
    # Simplified warehouse barriers (x0,y0,z0,x1,y1,z1).
    return [
        (4.0, 0.0, 0.0, 5.0, 6.0, 1.2),
        (8.0, 0.0, 0.0, 9.0, 6.0, 0.9),
        (12.0, 0.0, 0.0, 13.0, 3.0, 1.5),
        (12.0, 3.0, 0.0, 13.0, 6.0, 0.7),
    ]


def _ppp_occupancy(boxes, clearance: float = 0.45) -> KDTreeOccupancy:
    pts: list[list[float]] = []
    for x0, y0, z0, x1, y1, z1 in boxes:
        for x in np.linspace(x0, x1, 6):
            for y in np.linspace(y0, y1, 6):
                for z in np.linspace(z0, z1, 4):
                    pts.append([float(x), float(y), float(z)])
    return KDTreeOccupancy(np.array(pts, dtype=float), clearance=clearance)


def _ppp_reference_path() -> list[np.ndarray]:
    # Hand-crafted feasible path that climbs over / around barriers.
    waypoints = [
        (0.5, 3.0, 0.4),
        (3.0, 3.0, 0.4),
        (3.5, 3.0, 1.6),
        (5.5, 3.0, 1.6),
        (6.5, 3.0, 0.4),
        (7.5, 3.0, 1.4),
        (9.5, 3.0, 1.4),
        (10.5, 3.0, 0.4),
        (11.5, 1.5, 0.4),
        (11.5, 1.5, 1.8),
        (13.5, 1.5, 1.8),
        (14.5, 1.5, 0.4),
        (16.0, 3.0, 0.4),
    ]
    # Densify.
    dense: list[np.ndarray] = []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        for alpha in np.linspace(0.0, 1.0, 8, endpoint=False):
            p = (1 - alpha) * np.array(a) + alpha * np.array(b)
            dense.append(p)
    dense.append(np.array(waypoints[-1], dtype=float))
    return dense


def generate_ppp_video(out_path: Path, fps: int = 20) -> None:
    """PPP gantry EE tracking with JointSpaceMPC."""
    dt = 0.05
    boxes = _ppp_boxes()
    occ = _ppp_occupancy(boxes)
    path = _ppp_reference_path()
    arcs = [0.0]
    for i in range(1, len(path)):
        arcs.append(arcs[-1] + float(np.linalg.norm(path[i] - path[i - 1])))

    def path_at(s: float) -> np.ndarray:
        if s <= 0:
            return path[0].copy()
        if s >= arcs[-1]:
            return path[-1].copy()
        i = int(np.searchsorted(arcs, s) - 1)
        i = max(0, min(i, len(path) - 2))
        seg = arcs[i + 1] - arcs[i]
        alpha = 0.0 if seg < 1e-9 else (s - arcs[i]) / seg
        return (1 - alpha) * path[i] + alpha * path[i + 1]

    mpc = JointSpaceMPC(
        max_vel=np.array([2.0, 2.0, 1.5]),
        max_acc=np.array([4.0, 4.0, 3.0]),
        occupancy=occ,
        config=JointSpaceMPCConfig(
            horizon_step_count=12,
            dt=dt,
            weight_tracking=25.0,
            weight_obstacle=80.0,
            max_solver_iter_count=50,
        ),
    )
    mpc.reset(path[0].copy())
    race_speed = 1.2
    steps = int(18.0 / dt)
    qs = np.zeros((steps, 3))
    carrot_s = 0.0
    for i in range(steps):
        carrot_s = min(carrot_s + race_speed * dt, arcs[-1])
        # leash
        lag = np.linalg.norm(path_at(carrot_s) - mpc.q)
        if lag > 1.2:
            carrot_s = max(0.0, carrot_s - race_speed * dt)
        qs[i] = mpc.step(path_at(carrot_s), dt)

    fig = plt.figure(figsize=(8.5, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    for x0, y0, z0, x1, y1, z1 in boxes:
        ax.bar3d(
            x0,
            y0,
            z0,
            x1 - x0,
            y1 - y0,
            z1 - z0,
            color="#666666",
            alpha=0.35,
            shade=True,
        )
    pref = np.array(path)
    ax.plot(pref[:, 0], pref[:, 1], pref[:, 2], "--", color="#888888", lw=1.0)
    (trail,) = ax.plot([], [], [], color="#2a6f97", lw=2.0)
    (ee,) = ax.plot([], [], [], "o", color="#c45c26", ms=7)
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 6)
    ax.set_zlim(0, 2.5)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("PPP gantry — JointSpaceMPC carrot tracking")

    def update(frame: int):
        i = min(frame, steps - 1)
        trail.set_data(qs[: i + 1, 0], qs[: i + 1, 1])
        trail.set_3d_properties(qs[: i + 1, 2])
        ee.set_data([qs[i, 0]], [qs[i, 1]])
        ee.set_3d_properties([qs[i, 2]])
        return trail, ee

    _write_animation(fig, update, steps, out_path, fps)


# ---------------------------------------------------------------------------
# 3) RRP / SCARA-like arm
# ---------------------------------------------------------------------------


def _rrp_fk(q: np.ndarray, l1: float = 1.0, l2: float = 0.8) -> np.ndarray:
    q1, q2, z = float(q[0]), float(q[1]), float(q[2])
    x1 = l1 * math.cos(q1)
    y1 = l1 * math.sin(q1)
    x2 = x1 + l2 * math.cos(q1 + q2)
    y2 = y1 + l2 * math.sin(q1 + q2)
    return np.array(
        [
            [0.0, 0.0, z],
            [x1, y1, z],
            [x2, y2, z],
        ]
    )


def _rrp_joint_path() -> list[np.ndarray]:
    # Joint-space waypoints that swing around pillars then raise.
    raw = [
        (-2.4, 0.6, 0.2),
        (-1.8, 0.8, 0.2),
        (-1.0, 1.2, 0.2),
        (-0.2, 1.4, 0.3),
        (0.4, 1.2, 0.8),
        (1.0, 0.8, 1.2),
        (1.6, 0.4, 0.8),
        (2.0, 0.2, 0.3),
        (2.3, 0.1, 0.2),
    ]
    dense: list[np.ndarray] = []
    for a, b in zip(raw[:-1], raw[1:]):
        for alpha in np.linspace(0.0, 1.0, 10, endpoint=False):
            dense.append((1 - alpha) * np.array(a) + alpha * np.array(b))
    dense.append(np.array(raw[-1], dtype=float))
    return dense


def _rrp_occupancy() -> KDTreeOccupancy:
    # Sample C-space obstacles near pillar-blocked joint configs (approx).
    pts: list[list[float]] = []
    for q1 in np.linspace(-0.4, 0.4, 8):
        for q2 in np.linspace(0.8, 1.6, 8):
            for z in np.linspace(0.0, 0.5, 4):
                pts.append([float(q1), float(q2), float(z)])
    return KDTreeOccupancy(np.array(pts, dtype=float), clearance=0.25)


def generate_rrp_video(out_path: Path, fps: int = 20) -> None:
    """RRP arm FK visualization with JointSpaceMPC in joint space."""
    dt = 0.05
    path = _rrp_joint_path()
    occ = _rrp_occupancy()
    arcs = [0.0]
    for i in range(1, len(path)):
        arcs.append(arcs[-1] + float(np.linalg.norm(path[i] - path[i - 1])))

    def path_at(s: float) -> np.ndarray:
        if s <= 0:
            return path[0].copy()
        if s >= arcs[-1]:
            return path[-1].copy()
        i = int(np.searchsorted(arcs, s) - 1)
        i = max(0, min(i, len(path) - 2))
        seg = arcs[i + 1] - arcs[i]
        alpha = 0.0 if seg < 1e-9 else (s - arcs[i]) / seg
        return (1 - alpha) * path[i] + alpha * path[i + 1]

    mpc = JointSpaceMPC(
        max_vel=np.array([1.2, 1.2, 0.8]),
        max_acc=np.array([2.5, 2.5, 1.5]),
        occupancy=occ,
        config=JointSpaceMPCConfig(
            horizon_step_count=12,
            dt=dt,
            weight_tracking=22.0,
            weight_obstacle=70.0,
            max_solver_iter_count=50,
        ),
    )
    mpc.reset(path[0].copy())
    race_speed = 0.55
    steps = int(16.0 / dt)
    qs = np.zeros((steps, 3))
    carrot_s = 0.0
    for i in range(steps):
        carrot_s = min(carrot_s + race_speed * dt, arcs[-1])
        qs[i] = mpc.step(path_at(carrot_s), dt)

    fig = plt.figure(figsize=(8.0, 5.8))
    ax = fig.add_subplot(111, projection="3d")
    # Pillars in workspace.
    for cx, cy in ((0.75, -1.1), (-0.75, 1.1)):
        ax.bar3d(
            cx - 0.15,
            cy - 0.12,
            0.0,
            0.30,
            0.24,
            2.5,
            color="#555555",
            alpha=0.4,
        )
    (arm,) = ax.plot([], [], [], "-o", color="#c45c26", lw=2.5, ms=5)
    (trail,) = ax.plot([], [], [], color="#2a6f97", lw=1.5, alpha=0.8)
    ee_trail = np.array([_rrp_fk(q)[-1] for q in qs])
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_zlim(0.0, 2.8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("RRP / SCARA — JointSpaceMPC (FK view)")

    def update(frame: int):
        i = min(frame, steps - 1)
        links = _rrp_fk(qs[i])
        arm.set_data(links[:, 0], links[:, 1])
        arm.set_3d_properties(links[:, 2])
        trail.set_data(ee_trail[: i + 1, 0], ee_trail[: i + 1, 1])
        trail.set_3d_properties(ee_trail[: i + 1, 2])
        return arm, trail

    _write_animation(fig, update, steps, out_path, fps)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tools/output"),
        help="Directory for MP4 outputs",
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated subset: dubins,ppp,rrp",
    )
    args = parser.parse_args()
    only = {s.strip() for s in args.only.split(",") if s.strip()}

    jobs = [
        (
            "dubins",
            args.out_dir / "mpc_overview_dubins.mp4",
            generate_dubins_video,
        ),
        ("ppp", args.out_dir / "mpc_overview_ppp.mp4", generate_ppp_video),
        ("rrp", args.out_dir / "mpc_overview_rrp.mp4", generate_rrp_video),
    ]
    for name, path, fn in jobs:
        if only and name not in only:
            continue
        print(f"=== Generating {name} overview ===")
        fn(path, fps=args.fps)


if __name__ == "__main__":
    main()
