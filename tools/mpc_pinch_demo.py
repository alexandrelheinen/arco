#!/usr/bin/env python3
"""Headless pinch-corridor comparison: Pure Pursuit + APF vs path-following MPC.

Reproduces a warehouse-style pinch analogous to FRET ``str_002_col`` /
``str_008_col`` gaps and writes speed / lateral error / clearance plots to
``tools/output/mpc_pinch_demo.png``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from arco.control.mpc import (
    DubinsPathFollowingMPC,
    DubinsVehicleLimits,
    MPCTrackingLoop,
    PathFollowingMPCConfig,
)
from arco.control.pure_pursuit import PurePursuitController
from arco.control.tracking import TrackingLoop
from arco.guidance.vehicle import DubinsVehicle
from arco.mapping.kdtree import KDTreeOccupancy


def _pinch_occupancy(clearance: float = 0.30) -> KDTreeOccupancy:
    """Build two rectangular pillars forming a narrow gap on y=0.

    Gap width is ~0.40 m so the centerline sits inside the clearance
    margin (dist ≈ 0.20 m < clearance), forcing the tracker to slow
    and/or deviate — analogous to FRET warehouse pinch corridors.
    """
    pts: list[list[float]] = []
    for x in np.linspace(4.0, 5.5, 16):
        for y in np.linspace(-1.2, -0.20, 12):
            pts.append([float(x), float(y)])
        for y in np.linspace(0.20, 1.2, 12):
            pts.append([float(x), float(y)])
    return KDTreeOccupancy(np.array(pts, dtype=float), clearance=clearance)


def _straight_path(
    length: float = 12.0, step: float = 0.25
) -> list[tuple[float, float]]:
    xs = np.arange(0.0, length + step, step)
    return [(float(x), 0.0) for x in xs]


def _make_vehicle(cruise: float) -> DubinsVehicle:
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
    return vehicle


def _run_pp(
    path: list[tuple[float, float]],
    occupancy: KDTreeOccupancy,
    *,
    cruise: float,
    dt: float,
    duration: float,
    repulsion_gain: float,
) -> dict[str, np.ndarray]:
    vehicle = _make_vehicle(cruise)
    loop = TrackingLoop(
        vehicle,
        PurePursuitController(lookahead_distance=1.2),
        cruise_speed=cruise,
        occupancy=occupancy,
        repulsion_gain=repulsion_gain,
    )
    steps = int(duration / dt)
    speed = np.zeros(steps)
    lat = np.zeros(steps)
    clearance = np.zeros(steps)
    for i in range(steps):
        m = loop.step(path, dt=dt)
        speed[i] = m["speed"]
        lat[i] = abs(m["cross_track_error"])
        dist, _ = occupancy.nearest_obstacle(
            np.array(m["pose"][:2], dtype=float)
        )
        clearance[i] = dist
    return {
        "t": np.arange(steps) * dt,
        "speed": speed,
        "lat": lat,
        "clearance": clearance,
    }


def _run_mpc(
    path: list[tuple[float, float]],
    occupancy: KDTreeOccupancy,
    *,
    cruise: float,
    dt: float,
    duration: float,
) -> dict[str, np.ndarray]:
    vehicle = _make_vehicle(cruise)
    cfg = PathFollowingMPCConfig(
        horizon_step_count=24,
        dt=dt,
        cruise_speed=cruise,
        weight_contour=8.0,
        weight_heading=4.0,
        weight_progress=1.0,
        weight_control=0.05,
        weight_obstacle=120.0,
        obstacle_barrier_power=4.0,
        weight_terminal=20.0,
        max_solver_iter_count=80,
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
        occupancy=occupancy,
    )
    loop = MPCTrackingLoop(vehicle, tracker, cruise_speed=cruise)
    steps = int(duration / dt)
    speed = np.zeros(steps)
    lat = np.zeros(steps)
    clearance = np.zeros(steps)
    for i in range(steps):
        m = loop.step(path, dt=dt)
        speed[i] = m["speed"]
        lat[i] = abs(m["cross_track_error"])
        dist, _ = occupancy.nearest_obstacle(
            np.array(m["pose"][:2], dtype=float)
        )
        clearance[i] = dist
    return {
        "t": np.arange(steps) * dt,
        "speed": speed,
        "lat": lat,
        "clearance": clearance,
    }


def main() -> None:
    """Run the pinch demo and write the comparison figure."""
    cruise = 0.36
    dt = 0.05
    duration = 12.0
    clearance_margin = 0.30
    path = _straight_path()
    occupancy = _pinch_occupancy(clearance=clearance_margin)

    pp = _run_pp(
        path,
        occupancy,
        cruise=cruise,
        dt=dt,
        duration=duration,
        repulsion_gain=1.5,
    )
    mpc = _run_mpc(path, occupancy, cruise=cruise, dt=dt, duration=duration)

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.5), sharex=True)
    axes[0].plot(pp["t"], pp["speed"], label="PP+APF", color="#1f4e79")
    axes[0].plot(mpc["t"], mpc["speed"], label="MPC", color="#c45c26")
    axes[0].axhline(cruise, color="gray", ls="--", lw=0.8, label="cruise")
    axes[0].set_ylabel("speed (m/s)")
    axes[0].legend(loc="upper right")
    axes[0].set_title("Pinch corridor: PP+APF vs path-following MPC")

    axes[1].plot(pp["t"], pp["lat"], color="#1f4e79")
    axes[1].plot(mpc["t"], mpc["lat"], color="#c45c26")
    axes[1].set_ylabel("|lateral error| (m)")

    axes[2].plot(pp["t"], pp["clearance"], color="#1f4e79")
    axes[2].plot(mpc["t"], mpc["clearance"], color="#c45c26")
    axes[2].axhline(
        clearance_margin,
        color="gray",
        ls="--",
        lw=0.8,
        label="clearance margin",
    )
    axes[2].set_ylabel("min obstacle dist (m)")
    axes[2].set_xlabel("time (s)")
    axes[2].legend(loc="upper right")

    # Mark pinch entry.
    for ax in axes:
        ax.axvline(4.0 / max(cruise, 1e-6), color="#888888", ls=":", lw=0.8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mpc_pinch_demo.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Wrote {out_path}")
    print(
        f"PP min clearance={pp['clearance'].min():.3f} m, "
        f"MPC min clearance={mpc['clearance'].min():.3f} m"
    )
    print(
        f"MPC speed at pinch approach "
        f"(t={4.0 / cruise:.1f}s window): "
        f"{mpc['speed'][int((3.5 / cruise) / dt):int((5.0 / cruise) / dt)].min():.3f} m/s"
    )


if __name__ == "__main__":
    main()
