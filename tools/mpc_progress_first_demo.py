"""Compare stiff vs lane-aware progress-first contouring on a sharp corner.

Global planners ignore vehicle dynamics, so a polyline kink sharper than
``R_min = v / ω_max`` is not executable at cruise.  This demo shows how
the executed trajectory evolves under:

1. **Stiff** contouring (``deadzone=0``, low progress, high contour
   weight) — the classic fit that slows hard at infeasible corners.
2. **Lane-aware progress-first** (city defaults: small free band + lag) —
   the car slows / widens inside the lane while ``s`` advances — without
   treating half the road width as free space into the walls.

Output::

    tools/output/mpc_progress_first_demo.png

Usage::

    python tools/mpc_progress_first_demo.py
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from arco.control.mpc import (
    DubinsPathFollowingMPC,
    DubinsVehicleLimits,
    PathFollowingMPCConfig,
)
from arco.guidance.vehicle import DubinsVehicle


def _run(
    path: list[tuple[float, float]],
    cfg: PathFollowingMPCConfig,
    *,
    max_turn_rate: float,
    duration_s: float = 14.0,
) -> dict[str, np.ndarray]:
    """Simulate one tracker and return time series."""
    limits = DubinsVehicleLimits(
        max_speed=6.0,
        min_speed=0.0,
        max_turn_rate=max_turn_rate,
        max_acceleration=2.5,
        max_turn_rate_dot=math.radians(90.0),
    )
    mpc = DubinsPathFollowingMPC(vehicle_limits=limits, config=cfg)
    mpc.set_reference(path)
    vehicle = DubinsVehicle(
        x=path[0][0],
        y=path[0][1],
        heading=0.0,
        max_speed=limits.max_speed,
        min_speed=limits.min_speed,
        max_turn_rate=limits.max_turn_rate,
        max_acceleration=limits.max_acceleration,
        max_turn_rate_dot=limits.max_turn_rate_dot,
    )
    vehicle._speed = cfg.cruise_speed

    dt = cfg.dt
    n = int(duration_s / dt)
    xs = np.empty(n)
    ys = np.empty(n)
    progress = np.empty(n)
    lat = np.empty(n)
    for i in range(n):
        result = mpc.step(
            vehicle.pose,
            speed=vehicle.speed,
            turn_rate=vehicle.turn_rate,
            dt=dt,
        )
        vehicle.step(result.speed_cmd, result.turn_rate_cmd, dt)
        xs[i] = vehicle.x
        ys[i] = vehicle.y
        progress[i] = result.progress
        lat[i] = abs(result.cross_track_error)
    return {
        "t": np.arange(n) * dt,
        "x": xs,
        "y": ys,
        "progress": progress,
        "lat": lat,
    }


def main() -> Path:
    """Render the stiff vs progress-first comparison figure."""
    # Right-angle kink: infeasible if taken at cruise with R_min = v/ω.
    path = [
        (0.0, 0.0),
        (20.0, 0.0),
        (20.0, 20.0),
        (20.0, 40.0),
    ]
    cruise = 4.0
    omega = math.radians(30.0)
    r_min = cruise / omega

    stiff = PathFollowingMPCConfig(
        horizon_step_count=24,
        dt=0.05,
        cruise_speed=cruise,
        weight_contour=12.0,
        weight_heading=6.0,
        weight_progress=0.5,
        weight_lag=8.0,
        weight_control=0.05,
        weight_obstacle=0.0,
        weight_terminal=20.0,
        contour_deadzone=0.0,
        max_solver_iter_count=60,
    )
    progress_first = PathFollowingMPCConfig(
        horizon_step_count=24,
        dt=0.05,
        cruise_speed=cruise,
        weight_contour=8.0,
        weight_heading=4.0,
        weight_progress=4.0,
        weight_lag=4.0,
        weight_control=0.5,
        weight_obstacle=0.0,
        weight_terminal=8.0,
        contour_deadzone=1.2,  # scaled free band (~city 2.5 m)
        max_solver_iter_count=60,
    )

    a = _run(path, stiff, max_turn_rate=omega)
    b = _run(path, progress_first, max_turn_rate=omega)

    pref = np.asarray(path, dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), dpi=140)
    fig.patch.set_facecolor("#101418")
    for ax in axes:
        ax.set_facecolor("#101418")
        ax.tick_params(colors="#b0bec5")
        for spine in ax.spines.values():
            spine.set_color("#455a64")

    ax = axes[0]
    ax.plot(
        pref[:, 0],
        pref[:, 1],
        "--",
        color="#90a4ae",
        lw=1.5,
        label="plan (no dynamics)",
    )
    ax.plot(a["x"], a["y"], color="#ef5350", lw=2.0, label="stiff contouring")
    ax.plot(
        b["x"],
        b["y"],
        color="#26a69a",
        lw=2.2,
        label="lane-aware progress-first",
    )
    circ = plt.Circle(
        (20.0 - r_min, 0.0),
        r_min,
        fill=False,
        color="#ffcc80",
        ls=":",
        lw=1.0,
        label=f"R_min≈{r_min:.1f} m",
    )
    ax.add_patch(circ)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("XY: plan vs executed", color="white", fontsize=10)
    ax.legend(
        fontsize=7,
        facecolor="#1b2228",
        edgecolor="#455a64",
        labelcolor="white",
    )
    ax.set_xlabel("x [m]", color="#b0bec5")
    ax.set_ylabel("y [m]", color="#b0bec5")

    ax = axes[1]
    ax.plot(a["t"], a["progress"], color="#ef5350", lw=1.8, label="stiff s(t)")
    ax.plot(
        b["t"],
        b["progress"],
        color="#26a69a",
        lw=1.8,
        label="lane-aware s(t)",
    )
    ax.set_title("Arc-length progress", color="white", fontsize=10)
    ax.legend(
        fontsize=7,
        facecolor="#1b2228",
        edgecolor="#455a64",
        labelcolor="white",
    )
    ax.set_xlabel("t [s]", color="#b0bec5")
    ax.set_ylabel("s [m]", color="#b0bec5")

    ax = axes[2]
    ax.plot(a["t"], a["lat"], color="#ef5350", lw=1.8, label="stiff |e_lat|")
    ax.plot(
        b["t"],
        b["lat"],
        color="#26a69a",
        lw=1.8,
        label="lane-aware |e_lat|",
    )
    ax.axhline(1.2, color="#80cbc4", ls="--", lw=1.0, label="deadzone (demo)")
    ax.set_title("Lateral error", color="white", fontsize=10)
    ax.legend(
        fontsize=7,
        facecolor="#1b2228",
        edgecolor="#455a64",
        labelcolor="white",
    )
    ax.set_xlabel("t [s]", color="#b0bec5")
    ax.set_ylabel("|e_lat| [m]", color="#b0bec5")

    fig.suptitle(
        "Lane-aware progress-first: widen kinks inside the corridor, keep s advancing",
        color="white",
        fontsize=11,
    )
    fig.tight_layout()
    out_dir = Path(__file__).resolve().parents[1] / "tools" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mpc_progress_first_demo.png"
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)

    art = Path("/opt/cursor/artifacts")
    art.mkdir(parents=True, exist_ok=True)
    shutil.copy(out_path, art / "mpc_progress_first_demo.png")
    print(
        f"wrote {out_path}\n"
        f"stiff:   Δs={a['progress'][-1] - a['progress'][0]:.2f} m, "
        f"max|e_lat|={a['lat'].max():.2f} m\n"
        f"prog1st: Δs={b['progress'][-1] - b['progress'][0]:.2f} m, "
        f"max|e_lat|={b['lat'].max():.2f} m"
    )
    return out_path


if __name__ == "__main__":
    main()
