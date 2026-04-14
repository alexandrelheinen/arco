"""City race planning benchmark (RRT* vs SST) in 2-D.

Builds the procedural city-neighborhood scene and renders both planners
side-by-side using matplotlib on a light background.  For each planner
the figure shows four layers:

1. **Environment** — road dots (light) + obstacle KDTree points (red).
2. **Planned path** — raw discrete waypoints from the planner (dim).
3. **Predicted trajectory** — smoothed output of the TrajectoryOptimizer.
4. **Executed trajectory** — simulated vehicle following the predicted
   trajectory via :class:`~arco.control.tracking.TrackingLoop` (bright).

Usage
-----
Run interactively::

    python tools/examples/city.py

Save image::

    python tools/examples/city.py --save path/to/output.png
"""

from __future__ import annotations

import argparse
import logging
import math
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from arco.config.palette import (
    LAYER_ALPHA,
    annotation_hex,
    layer_hex,
    obstacle_hex,
)
from arco.tools.logging_config import configure_logging
from arco.tools.simulator.scenes.sparse import CityScene
from arco.tools.simulator.sim.tracking import build_vehicle_sim

logger = logging.getLogger(__name__)


def _polyline_length(path: list[np.ndarray] | None) -> float:
    if path is None or len(path) < 2:
        return 0.0
    return sum(
        float(np.linalg.norm(path[i + 1] - path[i]))
        for i in range(len(path) - 1)
    )


def _simulate_trajectory(
    traj: list[np.ndarray] | None,
    scene: CityScene,
    dt: float = 0.05,
) -> list[tuple[float, float]]:
    """Run a headless TrackingLoop simulation and return executed positions.

    Args:
        traj: Optimised trajectory states (each element is at least (x, y, …)).
        scene: Built :class:`~arco.tools.simulator.scenes.sparse.CityScene`
            providing the vehicle config and occupancy map.
        dt: Simulation time step in seconds.

    Returns:
        List of ``(x, y)`` positions recorded at every control step.
        Empty list when *traj* is ``None`` or too short.
    """
    if traj is None or len(traj) < 2:
        return []
    v_cfg = scene.vehicle_config  # noqa: SLF001
    occ = scene._occ  # noqa: SLF001
    waypoints: list[tuple[float, float]] = [
        (float(p[0]), float(p[1])) for p in traj
    ]
    _, loop = build_vehicle_sim(waypoints, v_cfg, occupancy=occ)
    executed: list[tuple[float, float]] = [waypoints[0]]
    max_steps = max(3000, len(waypoints) * 300)
    gx, gy = waypoints[-1]
    for _ in range(max_steps):
        result = loop.step(waypoints, dt)
        x, y, _ = result["pose"]
        executed.append((x, y))
        if math.hypot(x - gx, y - gy) < v_cfg.goal_radius:
            break
    return executed


def _draw_panel(
    ax: plt.Axes,
    scene: CityScene,
    title: str,
    path: list[np.ndarray] | None,
    traj: list[np.ndarray] | None,
    executed: list[tuple[float, float]],
    color: str,
    traj_color: str,
    vehicle_color: str,
    metrics: dict[str, float | int | str],
) -> None:
    occ = scene._occ  # noqa: SLF001 - example-only visualization access.
    road = np.array(scene.road_dots) if scene.road_dots else np.empty((0, 2))
    obs = np.array(occ.points) if occ is not None else np.empty((0, 2))

    if len(road) > 0:
        ax.scatter(road[:, 0], road[:, 1], s=2, c="#b8c4ce", alpha=0.6)
    if len(obs) > 0:
        ax.scatter(obs[:, 0], obs[:, 1], s=3, c=obstacle_hex(), alpha=0.55)

    if path is not None and len(path) >= 2:
        arr = np.array(path)
        ax.plot(
            arr[:, 0],
            arr[:, 1],
            color=color,
            linewidth=1.6,
            alpha=0.45,
            label="Planned path",
        )
    if traj is not None and len(traj) >= 2:
        arr = np.array(traj)
        ax.plot(
            arr[:, 0],
            arr[:, 1],
            color=traj_color,
            linewidth=2.2,
            label="Predicted traj",
        )
    if len(executed) >= 2:
        ex = np.array(executed)
        ax.plot(
            ex[:, 0],
            ex[:, 1],
            color=vehicle_color,
            linewidth=1.8,
            linestyle="--",
            alpha=0.85,
            label="Executed traj",
        )

    sx, sy = scene._start  # noqa: SLF001
    gx, gy = scene._goal  # noqa: SLF001
    ann = annotation_hex()
    ax.plot(sx, sy, "s", color=ann, ms=8, label="Start")
    ax.plot(gx, gy, "x", color=ann, ms=8, mew=2, label="Goal")

    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")

    lines = [
        f"Steps/nodes: {metrics['steps']}/{metrics['nodes']}",
        f"Planner time: {metrics['planner_time']:.1f}s",
        f"Path length: {float(metrics['planned_path_length']):.1f} m",
        f"Traj length: {float(metrics['trajectory_arc_length']):.1f} m",
    ]
    if len(executed) >= 2:
        exec_len = sum(
            math.hypot(
                executed[i + 1][0] - executed[i][0],
                executed[i + 1][1] - executed[i][1],
            )
            for i in range(len(executed) - 1)
        )
        lines.append(f"Executed length: {exec_len:.1f} m")
    lines.append(f"Status: {metrics['path_status']}")
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="black",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
    )


def main(cfg: dict, save_path: str | None = None) -> None:
    if save_path is not None:
        matplotlib.use("Agg")

    scene = CityScene(cfg.get("planner", {}), cfg.get("world", {}))
    scene.build()

    rrt_path = [np.asarray(p) for p in (scene._rrt_path or [])]  # noqa: SLF001
    sst_path = [np.asarray(p) for p in (scene._sst_path or [])]  # noqa: SLF001
    rrt_traj = [
        np.asarray(p) for p in (scene._rrt_traj_states or [])
    ]  # noqa: SLF001
    sst_traj = [
        np.asarray(p) for p in (scene._sst_traj_states or [])
    ]  # noqa: SLF001

    logger.info("Simulating RRT* executed trajectory …")
    rrt_executed = _simulate_trajectory(rrt_traj or rrt_path, scene)
    logger.info("Simulating SST executed trajectory …")
    sst_executed = _simulate_trajectory(sst_traj or sst_path, scene)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor("white")

    _draw_panel(
        ax1,
        scene,
        "City race - RRT*",
        rrt_path,
        rrt_traj,
        rrt_executed,
        color=layer_hex("rrt", "path"),
        traj_color=layer_hex("rrt", "trajectory"),
        vehicle_color=layer_hex("rrt", "vehicle"),
        metrics=scene.rrt_metrics,
    )
    _draw_panel(
        ax2,
        scene,
        "City race - SST",
        sst_path,
        sst_traj,
        sst_executed,
        color=layer_hex("sst", "path"),
        traj_color=layer_hex("sst", "trajectory"),
        vehicle_color=layer_hex("sst", "vehicle"),
        metrics=scene.sst_metrics,
    )

    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=7)

    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=140)
        logger.info("Saved city planning example to %s", save_path)
    else:
        plt.show()


if __name__ == "__main__":
    import yaml as _yaml

    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scenario", metavar="FILE", help="Path to scenario YAML file."
    )
    parser.add_argument("--save", metavar="PATH", default=None)
    args = parser.parse_args()
    with open(args.scenario) as _fh:
        _cfg = _yaml.safe_load(_fh) or {}
    main(_cfg, save_path=args.save)
