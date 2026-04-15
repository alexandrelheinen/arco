"""Object-centric control — RRT* vs SST in (x, y, ψ) C-space.

Visualises the planning results for the piano movers problem: a 2-D rigid
body (square or circle) manipulated from pose A to pose B by N mobile
actuators.  Each planner's output is pruned and trajectory-optimised;
both the raw path and the optimised trajectory are overlaid.

Figure layout (standard two-frame)
------------------------------------
* Top-left  — Workspace: Cartesian 2-D view with rectangular obstacles,
  start/goal poses, both planners' paths and trajectories.
* Top-right — C-space (x m, y m): collision projection with both planners'
  joint paths/trajectories.  The (x, ψ) slice is dropped in favour of
  showing the single most informative 2-D C-space projection.
* Bottom    — Metrics: per-planner information and optimiser status.

Usage
-----
Run interactively (opens a matplotlib window)::

    python tools/examples/occ.py

Save the output image without opening a window::

    python tools/examples/occ.py --save path/to/output.png
"""

from __future__ import annotations

import argparse
import logging
import time

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from arco.tools.logging_config import configure_logging
from arco.tools.simulator.scenes.occ import OCCScene
from arco.tools.viewer import FrameRenderer, SceneSnapshot
from arco.tools.viewer.layout import StandardLayout
from arco.tools.viewer.utils import format_clock

logger = logging.getLogger(__name__)


def _build_occ_snapshot(
    planner: str,
    path: list[np.ndarray] | None,
    raw_path: list[np.ndarray] | None,
    traj: list[np.ndarray] | None,
    collision_pts: list[list[float]],
    start_pose: np.ndarray,
    goal_pose: np.ndarray,
    *,
    include_obstacles: bool = True,
) -> SceneSnapshot:
    """Build a SceneSnapshot from OCC planning results (x,y 2-D projection).

    Args:
        planner: Planner key, e.g. ``"rrt"`` or ``"sst"``.
        path: Pruned path in (x, y, ψ) space.
        raw_path: Raw planner path before pruning.
        traj: Optimised trajectory states.
        collision_pts: C-space collision samples ``[[x, y, ψ], …]``.
        start_pose: Start pose array ``[x, y, ψ]``.
        goal_pose: Goal pose array ``[x, y, ψ]``.
        include_obstacles: When ``False`` the obstacles list is left empty
            (avoid re-rendering on overlaid planners).

    Returns:
        A :class:`~arco.tools.viewer.SceneSnapshot` with 2-D (x, y) coords.
    """
    obs: list[list[float]] = (
        [[float(p[0]), float(p[1])] for p in collision_pts]
        if include_obstacles
        else []
    )
    return SceneSnapshot.from_planning_result(
        scenario="occ",
        planner=planner,
        start=[float(start_pose[0]), float(start_pose[1])],
        goal=[float(goal_pose[0]), float(goal_pose[1])],
        obstacles=obs,
        found_path=(
            [[float(p[0]), float(p[1])] for p in raw_path]
            if raw_path
            else None
        ),
        pruned_path=(
            [[float(p[0]), float(p[1])] for p in path] if path else None
        ),
        adjusted_trajectory=(
            [[float(p[0]), float(p[1])] for p in traj] if traj else None
        ),
    )


def main(cfg: dict, save_path: str | None = None) -> None:
    """Run the OCC example visualization.

    Args:
        cfg: Scenario configuration dictionary (loaded from ``occ.yml``).
        save_path: Optional path to save the output image.  When ``None``
            an interactive matplotlib window is opened instead.
    """
    configure_logging()
    if save_path is not None:
        matplotlib.use("Agg")

    env_cfg: dict = cfg.get("environment", {})

    t_start = time.perf_counter()
    scene = OCCScene(cfg)
    scene.build()
    elapsed = time.perf_counter() - t_start
    logger.info("Scene built in %s", format_clock(elapsed))

    collision_pts = scene.collision_pts
    rrt_path = scene.rrt_path
    sst_path = scene.sst_path
    rrt_traj = scene.rrt_traj
    sst_traj = scene.sst_traj
    start_pose = scene.start_pose
    goal_pose = scene.goal_pose
    obstacles = scene.obstacles
    x_range = [float(v) for v in env_cfg.get("x_range", [-4, 4])]
    y_range = [float(v) for v in env_cfg.get("y_range", [-3, 3])]

    rrt_snap = _build_occ_snapshot(
        "rrt",
        rrt_path,
        scene.rrt_raw_path,
        rrt_traj,
        collision_pts,
        start_pose,
        goal_pose,
        include_obstacles=True,
    )
    sst_snap = _build_occ_snapshot(
        "sst",
        sst_path,
        scene.sst_raw_path,
        sst_traj,
        collision_pts,
        start_pose,
        goal_pose,
        include_obstacles=False,
    )

    fig, ax_ws, ax_cs, ax_bottom = StandardLayout.create(
        title="OCC — Object-centric control (piano movers)"
    )

    # ---- ax_ws: Cartesian 2-D workspace ------------------------------------
    for obs in obstacles:
        xmin, ymin, xmax, ymax = obs
        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=1,
            edgecolor="black",
            facecolor="#c05050",
            alpha=0.7,
        )
        ax_ws.add_patch(rect)
    FrameRenderer(draw_tree=False, draw_obstacles=False).render(
        ax_ws, rrt_snap
    )
    FrameRenderer(
        draw_tree=False, draw_obstacles=False, draw_start_goal=False
    ).render(ax_ws, sst_snap)
    ax_ws.set_xlim(x_range)
    ax_ws.set_ylim(y_range)
    ax_ws.set_title("Workspace (Cartesian x–y)")
    ax_ws.set_xlabel("x (m)")
    ax_ws.set_ylabel("y (m)")
    ax_ws.legend(fontsize=8)
    ax_ws.set_aspect("equal")
    ax_ws.grid(True, alpha=0.3)

    # ---- ax_cs: C-space (x, y) projection ---------------------------------
    FrameRenderer(draw_tree=False).render(ax_cs, rrt_snap)
    FrameRenderer(
        draw_tree=False, draw_obstacles=False, draw_start_goal=False
    ).render(ax_cs, sst_snap)
    ax_cs.set_xlim(x_range)
    ax_cs.set_ylim(y_range)
    ax_cs.set_title("C-space: (x, y) projection")
    ax_cs.set_xlabel("x (m)")
    ax_cs.set_ylabel("y (m)")
    ax_cs.legend(fontsize=8)
    ax_cs.set_aspect("equal")
    ax_cs.grid(True, alpha=0.3)

    # ---- Bottom: metrics ---------------------------------------------------
    rrt_pts = len(rrt_path) if rrt_path else 0
    sst_pts = len(sst_path) if sst_path else 0
    rrt_traj_pts = len(rrt_traj) if rrt_traj else 0
    sst_traj_pts = len(sst_traj) if sst_traj else 0
    StandardLayout.write_metrics(
        ax_bottom,
        [
            f"Build time: {format_clock(elapsed)} | "
            f"Collision samples: {len(collision_pts)}",
            f"RRT*  path waypoints: {rrt_pts} | "
            f"traj waypoints: {rrt_traj_pts}",
            f"SST   path waypoints: {sst_pts} | "
            f"traj waypoints: {sst_traj_pts}",
        ],
        columns=1,
    )

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved figure to %s", save_path)
    else:
        plt.show()


if __name__ == "__main__":
    import yaml as _yaml

    _p = argparse.ArgumentParser(
        description="OCC planning visualisation (RRT* vs SST)"
    )
    _p.add_argument(
        "scenario", metavar="FILE", help="Path to scenario YAML file."
    )
    _p.add_argument(
        "--save", metavar="PATH", default=None, help="Save figure to PATH."
    )
    _args = _p.parse_args()
    with open(_args.scenario) as _fh:
        _cfg = _yaml.safe_load(_fh) or {}
    main(_cfg, save_path=_args.save)
