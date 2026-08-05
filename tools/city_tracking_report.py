#!/usr/bin/env python
"""Headless closed-loop tracking report for the city race scenario.

Rebuilds the exact ``map/city*.yml`` scenario (world, planners, optimizer)
without any OpenGL window, runs the same MPC tracking loops the recorded
race uses, and reports per-racer quality metrics:

* finish time (or timeout),
* mean / max absolute cross-track error,
* minimum clearance to the building point cloud (collision gate),
* solver failure count.

A matplotlib overview image (map + executed trajectories) is written next
to the report so tracking quality can be inspected visually without
recording a full video.

Usage
-----
::

    python tools/city_tracking_report.py                       # preview map
    python tools/city_tracking_report.py --map map/city.yml    # full budgets
    python tools/city_tracking_report.py --out /tmp/report.png
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import yaml

from arco.simulator.scenes.sparse import CityScene
from arco.simulator.sim.city_race_style import (
    DEFAULT_CITY_HORIZON_DT,
    DEFAULT_CITY_HORIZON_STEP_COUNT,
    VEH_HALF_L,
    VEH_HALF_W,
)
from arco.simulator.sim.tracking import (
    build_vehicle_mpc_sim,
    build_vehicle_sim,
    path_following_mpc_config_from_simulator,
)

# Simulation timestep — matches src/arco/config/simulator.yml.
_SIM_DT = 0.1
# Wall-clock cap per racer (simulated seconds).
_TIMEOUT_S = 150.0
# Extra margin (m) around the rendered vehicle rectangle when testing
# building-point overlap (accounts for the drawn point size).
_FOOTPRINT_MARGIN_M = 0.5


def _footprint_overlap(
    x: float,
    y: float,
    heading: float,
    obstacle_xy: np.ndarray,
) -> bool:
    """Whether an obstacle point lies inside the rendered car rectangle.

    Args:
        x: Vehicle center x (m).
        y: Vehicle center y (m).
        heading: Vehicle heading (rad).
        obstacle_xy: Nearest obstacle point ``[x, y]``.

    Returns:
        ``True`` when the point falls inside the oriented glyph footprint
        (plus :data:`_FOOTPRINT_MARGIN_M`), i.e. a visual collision.
    """
    dx = float(obstacle_xy[0]) - x
    dy = float(obstacle_xy[1]) - y
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    lx = dx * cos_h + dy * sin_h
    ly = -dx * sin_h + dy * cos_h
    return (
        abs(lx) <= VEH_HALF_L + _FOOTPRINT_MARGIN_M
        and abs(ly) <= VEH_HALF_W + _FOOTPRINT_MARGIN_M
    )


@dataclass
class RacerReport:
    """Aggregated closed-loop metrics for one racer.

    Attributes:
        name: Planner name (``RRT*`` / ``SST`` / ``A*``).
        finish_time: Simulated seconds to reach the goal, ``None`` on timeout.
        max_abs_lat: Maximum absolute cross-track error (m).
        mean_abs_lat: Mean absolute cross-track error (m).
        min_clearance: Minimum distance to the building point cloud (m).
        solver_failures: Number of failed MPC solves.
        steps: Number of executed simulation steps.
        trajectory: Executed ``(x, y)`` positions.
        collision_count: Steps with a building point inside the rendered
            vehicle footprint.
    """

    name: str
    finish_time: float | None = None
    max_abs_lat: float = 0.0
    mean_abs_lat: float = 0.0
    min_clearance: float = float("inf")
    solver_failures: int = 0
    steps: int = 0
    trajectory: list[tuple[float, float]] = field(default_factory=list)
    collision_count: int = 0

    @property
    def collided(self) -> bool:
        """Whether the racer visually clipped a building."""
        return self.collision_count > 0


def _run_racer(
    name: str,
    waypoints: list[tuple[float, float]],
    scene: CityScene,
    tracker_mode: str,
) -> RacerReport:
    """Run one racer's closed loop to the goal and collect metrics.

    Args:
        name: Display name for the report.
        waypoints: Optimized reference waypoints from the scene.
        scene: Built city scene (provides occupancy + vehicle config).
        tracker_mode: ``"mpc"`` or ``"pure_pursuit"``.

    Returns:
        Populated :class:`RacerReport`.
    """
    report = RacerReport(name=name)
    if not waypoints:
        return report
    cfg = scene.vehicle_config
    occ = scene._occ
    if tracker_mode == "mpc":
        mpc_cfg = path_following_mpc_config_from_simulator(
            scene._sim_cfg,
            default_horizon_step_count=DEFAULT_CITY_HORIZON_STEP_COUNT,
            default_horizon_dt=DEFAULT_CITY_HORIZON_DT,
        )
        vehicle, loop = build_vehicle_mpc_sim(waypoints, cfg, mpc_cfg, occ)
    else:
        vehicle, loop = build_vehicle_sim(waypoints, cfg, occ)

    gx, gy = waypoints[-1]
    abs_lats: list[float] = []
    t = 0.0
    while t < _TIMEOUT_S:
        metrics = loop.step(waypoints, dt=_SIM_DT)
        t += _SIM_DT
        report.steps += 1
        report.trajectory.append((vehicle.x, vehicle.y))
        lat = abs(float(metrics.get("cross_track_error", 0.0)))
        abs_lats.append(lat)
        report.max_abs_lat = max(report.max_abs_lat, lat)
        if metrics.get("mpc_solver_success") is False:
            report.solver_failures += 1
        dist, nearest = occ.nearest_obstacle(
            np.array([vehicle.x, vehicle.y], dtype=float)
        )
        report.min_clearance = min(report.min_clearance, float(dist))
        if _footprint_overlap(vehicle.x, vehicle.y, vehicle.heading, nearest):
            report.collision_count += 1
        if math.hypot(vehicle.x - gx, vehicle.y - gy) < cfg.goal_radius:
            report.finish_time = t
            break
    report.mean_abs_lat = float(np.mean(abs_lats)) if abs_lats else 0.0
    return report


def _save_overview(
    scene: CityScene, reports: list[RacerReport], out_path: str
) -> None:
    """Save a map + executed-trajectory overview image.

    Args:
        scene: Built city scene.
        reports: Per-racer reports with executed trajectories.
        out_path: PNG output path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10))
    pts = np.asarray(scene._occ.points)
    ax.scatter(pts[:, 0], pts[:, 1], s=1.0, c="#666666", label="buildings")
    colors = {"RRT*": "#3b82f6", "SST": "#22c55e", "A*": "#a855f7"}
    refs = {
        "RRT*": scene.rrt_waypoints,
        "SST": scene.sst_waypoints,
        "A*": scene.astar_waypoints,
    }
    for rep in reports:
        color = colors.get(rep.name, "#ffffff")
        ref = refs.get(rep.name) or []
        if ref:
            rx = [p[0] for p in ref]
            ry = [p[1] for p in ref]
            ax.plot(rx, ry, color=color, lw=0.8, alpha=0.35)
        if rep.trajectory:
            tx = [p[0] for p in rep.trajectory]
            ty = [p[1] for p in rep.trajectory]
            ax.plot(tx, ty, color=color, lw=1.8, label=f"{rep.name} executed")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("City race — executed trajectories vs references")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Optional CLI argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: 0 when every racer finishes with no collision.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--map", default="map/city_mpc_preview.yml", help="Scenario YAML"
    )
    parser.add_argument(
        "--out", default="/tmp/city_tracking_report.png", help="Overview PNG"
    )
    args = parser.parse_args(argv)

    with open(args.map, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    scene = CityScene(
        cfg.get("planner", {}),
        cfg.get("world", {}),
        sim_cfg=cfg.get("simulator", {}),
    )
    t0 = time.perf_counter()
    scene.build()
    print(f"Scene built in {time.perf_counter() - t0:.1f} s")

    tracker_mode = str(cfg.get("simulator", {}).get("tracker", "pure_pursuit"))
    reports: list[RacerReport] = []
    for name, wps in (
        ("RRT*", scene.rrt_waypoints),
        ("SST", scene.sst_waypoints),
        ("A*", scene.astar_waypoints),
    ):
        t0 = time.perf_counter()
        rep = _run_racer(name, wps, scene, tracker_mode)
        wall = time.perf_counter() - t0
        reports.append(rep)
        finish = (
            f"{rep.finish_time:.1f} s"
            if rep.finish_time is not None
            else "TIMEOUT"
        )
        print(
            f"{name:5s} finish={finish:>9s}  "
            f"lat(mean/max)={rep.mean_abs_lat:.2f}/{rep.max_abs_lat:.2f} m  "
            f"min_clear={rep.min_clearance:.2f} m  "
            f"fails={rep.solver_failures}/{rep.steps}  "
            f"wall={wall:.1f} s  "
            f"{'COLLISION x' + str(rep.collision_count) if rep.collided else 'clean'}"
        )

    _save_overview(scene, reports, args.out)
    print(f"Overview image: {args.out}")

    ok = all(
        (r.finish_time is not None) and not r.collided
        for r in reports
        if r.steps > 0
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
