"""City race planning benchmark (RRT* vs SST) in 2-D.

Builds the procedural city-neighborhood scene and renders both planners
on a common figure using the standard two-frame layout.  Both planners'
results are overlaid on the same axes for direct comparison.

Figure layout (standard two-frame)
------------------------------------
* Top-left  — Workspace: road network + obstacle cloud + both planners'
  paths, trajectories, and executed traces overlaid.
* Top-right — C-space: for the city scenario the C-space equals the
  workspace (2-D Dubins position).  Shown with a clearance heatmap.
* Bottom    — Metrics: per-planner step counts, planning times, path and
  trajectory lengths.

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

from arco.tools.logging_config import configure_logging
from arco.tools.simulator.scenes.sparse import CityScene
from arco.tools.simulator.sim.tracking import build_vehicle_sim
from arco.tools.viewer import FrameRenderer, SceneSnapshot
from arco.tools.viewer.layout import StandardLayout
from arco.tools.viewer.utils import polyline_length

logger = logging.getLogger(__name__)


def _build_city_snapshot(
    planner: str,
    scene: CityScene,
    path: list[np.ndarray],
    raw_path: list[np.ndarray] | None,
    traj: list[np.ndarray],
    executed: list[tuple[float, float]],
    *,
    include_obstacles: bool = True,
) -> SceneSnapshot:
    """Build a SceneSnapshot from city planning results.

    Args:
        planner: Planner key, e.g. ``"rrt"`` or ``"sst"``.
        scene: Built :class:`~arco.tools.simulator.scenes.sparse.CityScene`.
        path: Pruned planned path.
        raw_path: Raw planner path before pruning.
        traj: Optimised trajectory states.
        executed: Executed ``(x, y)`` positions from the tracking loop.
        include_obstacles: When ``False`` the obstacles list is left empty
            (avoid re-rendering on overlaid planners).

    Returns:
        A :class:`~arco.tools.viewer.SceneSnapshot` for the given planner.
    """
    occ = scene._occ  # noqa: SLF001
    obs: list[list[float]] = (
        [[float(v) for v in pt] for pt in occ.points]
        if include_obstacles and occ is not None
        else []
    )
    sx, sy = scene._start  # noqa: SLF001
    gx, gy = scene._goal  # noqa: SLF001
    return SceneSnapshot.from_planning_result(
        scenario="city",
        planner=planner,
        start=[float(sx), float(sy)],
        goal=[float(gx), float(gy)],
        obstacles=obs,
        found_path=(
            [[float(v) for v in p] for p in raw_path] if raw_path else None
        ),
        pruned_path=(
            [[float(v) for v in p] for p in path] if path else None
        ),
        adjusted_trajectory=(
            [[float(v) for v in p] for p in traj] if traj else None
        ),
        executed_trajectory=(
            [[float(x), float(y)] for x, y in executed]
            if len(executed) >= 2
            else None
        ),
    )


def _simulate_trajectory(
    traj: list[np.ndarray] | None,
    scene: CityScene,
    dt: float = 0.05,
) -> list[tuple[float, float]]:
    """Run a headless TrackingLoop simulation and return executed positions.

    Args:
        traj: Optimised trajectory states (each element is at least
            (x, y, …)).
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




def main(cfg: dict, save_path: str | None = None) -> None:
    """Run the city benchmark and display or save the figure.

    Args:
        cfg: Scenario configuration dictionary (loaded from ``city.yml``).
        save_path: If provided, save the figure to this path instead of
            opening an interactive window.
    """
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

    rrt_snap = _build_city_snapshot(
        "rrt", scene, rrt_path, scene.rrt_raw_path, rrt_traj, rrt_executed,
        include_obstacles=True,
    )
    sst_snap = _build_city_snapshot(
        "sst", scene, sst_path, scene.sst_raw_path, sst_traj, sst_executed,
        include_obstacles=False,
    )

    fig, ax_ws, ax_cs, ax_bottom = StandardLayout.create(
        title="City race benchmark — RRT* vs SST"
    )

    # ---- ax_ws: workspace — both planners overlaid -------------------------
    occ = scene._occ  # noqa: SLF001
    road = (
        np.array(scene.road_dots) if scene.road_dots else np.empty((0, 2))
    )
    if len(road) > 0:
        ax_ws.scatter(road[:, 0], road[:, 1], s=2, c="#b8c4ce", alpha=0.6)
    FrameRenderer(draw_tree=False).render(ax_ws, rrt_snap)
    FrameRenderer(
        draw_tree=False, draw_obstacles=False, draw_start_goal=False
    ).render(ax_ws, sst_snap)
    ax_ws.set_title("Workspace")
    ax_ws.set_xlabel("X (m)")
    ax_ws.set_ylabel("Y (m)")
    ax_ws.set_aspect("equal")
    ax_ws.grid(True, alpha=0.3)
    ax_ws.legend(loc="upper right", fontsize=7)

    # ---- ax_cs: C-space = workspace for 2-D Dubins -------------------------
    # Show clearance heatmap (distance field) as background.
    if occ is not None:
        planner_cfg = cfg.get("planner", {})
        bounds = planner_cfg.get("bounds", [[0, 200], [0, 200]])
        x_min, x_max = float(bounds[0][0]), float(bounds[0][1])
        y_min, y_max = float(bounds[1][0]), float(bounds[1][1])
        resolution = 120
        xs = np.linspace(x_min, x_max, resolution)
        ys = np.linspace(y_min, y_max, resolution)
        xx, yy = np.meshgrid(xs, ys)
        grid_pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
        distances = occ.query_distances(grid_pts)
        dist_img = distances.reshape(resolution, resolution)
        ax_cs.imshow(
            dist_img,
            origin="lower",
            extent=[x_min, x_max, y_min, y_max],
            cmap="Greys_r",
            vmin=0,
            vmax=float(np.percentile(distances, 80)),
            aspect="auto",
            alpha=0.5,
        )
    FrameRenderer(draw_tree=False, draw_obstacles=False).render(ax_cs, rrt_snap)
    FrameRenderer(
        draw_tree=False, draw_obstacles=False, draw_start_goal=False
    ).render(ax_cs, sst_snap)
    ax_cs.set_title("C-space (x m, y m)")
    ax_cs.set_xlabel("X (m)")
    ax_cs.set_ylabel("Y (m)")
    ax_cs.set_aspect("equal")
    ax_cs.grid(True, alpha=0.3)
    ax_cs.legend(loc="upper right", fontsize=7)

    # ---- Bottom: metrics ---------------------------------------------------
    m_rrt = scene.rrt_metrics
    m_sst = scene.sst_metrics

    def _exec_len(pts: list[tuple[float, float]]) -> float:
        if len(pts) < 2:
            return 0.0
        return sum(
            math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
            for i in range(len(pts) - 1)
        )

    StandardLayout.write_metrics(
        ax_bottom,
        [
            f"RRT*  steps/nodes: {m_rrt['steps']}/{m_rrt['nodes']} | "
            f"time: {float(m_rrt['planner_time']):.1f} s | "
            f"path: {float(m_rrt['planned_path_length']):.1f} m | "
            f"traj: {float(m_rrt['trajectory_arc_length']):.1f} m | "
            f"executed: {_exec_len(rrt_executed):.1f} m | "
            f"status: {m_rrt['path_status']}",
            f"SST   steps/nodes: {m_sst['steps']}/{m_sst['nodes']} | "
            f"time: {float(m_sst['planner_time']):.1f} s | "
            f"path: {float(m_sst['planned_path_length']):.1f} m | "
            f"traj: {float(m_sst['trajectory_arc_length']):.1f} m | "
            f"executed: {_exec_len(sst_executed):.1f} m | "
            f"status: {m_sst['path_status']}",
        ],
    )

    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
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
