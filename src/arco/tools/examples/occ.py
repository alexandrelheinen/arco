"""Object-centric control — RRT* vs SST in (x, y, ψ) C-space.

Visualises the planning results for the piano movers problem: a 2-D rigid
body (square or circle) manipulated from pose A to pose B by N mobile
actuators.

Figure layout (3 subplots in a row)
-------------------------------------
* Left   — C-space (x, y) slice with collision occupancy + RRT* path
* Middle — C-space (x, ψ) slice with collision occupancy + RRT* path
* Right  — Cartesian 2-D view with obstacles, start/goal poses, and paths

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
import math
import time

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from arco.config.palette import annotation_hex, obstacle_hex
from arco.tools.logging_config import configure_logging
from arco.tools.simulator.scenes.occ import OCCScene

logger = logging.getLogger(__name__)


def _format_clock(seconds: float) -> str:
    """Format seconds as ``MMminSSs``.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted string.
    """
    rounded = int(round(seconds))
    mins, secs = divmod(rounded, 60)
    return f"{mins:02d}min{secs:02d}s"


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
    logger.info("Scene built in %s", _format_clock(elapsed))

    collision_pts = scene.collision_pts
    rrt_path = scene.rrt_path
    sst_path = scene.sst_path
    start_pose = scene.start_pose
    goal_pose = scene.goal_pose
    obstacles = scene.obstacles
    x_range = [float(v) for v in env_cfg.get("x_range", [-4, 4])]
    y_range = [float(v) for v in env_cfg.get("y_range", [-3, 3])]

    # ---------------------------------------------------------------------------
    # Figure
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("OCC — Object-centric control (piano movers)", fontsize=14)

    # -- Subplot 1: C-space (x, y) projection --
    ax1 = axes[0]
    ax1.set_title("C-space: (x, y) projection")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    if collision_pts:
        cxs = [p[0] for p in collision_pts]
        cys = [p[1] for p in collision_pts]
        ax1.scatter(
            cxs, cys, s=2, c=obstacle_hex(), alpha=0.3, label="Collision"
        )
    if rrt_path and len(rrt_path) >= 2:
        rxs = [p[0] for p in rrt_path]
        rys = [p[1] for p in rrt_path]
        ax1.plot(rxs, rys, "b-", linewidth=2, label="RRT*")
    if sst_path and len(sst_path) >= 2:
        sxs = [p[0] for p in sst_path]
        sys_ = [p[1] for p in sst_path]
        ax1.plot(sxs, sys_, "g--", linewidth=2, label="SST")
    ax1.plot(
        start_pose[0],
        start_pose[1],
        "s",
        color=annotation_hex(),
        markersize=10,
        label="Start",
    )
    ax1.plot(
        goal_pose[0],
        goal_pose[1],
        "x",
        color=annotation_hex(),
        markersize=12,
        label="Goal",
    )
    ax1.set_xlim(x_range)
    ax1.set_ylim(y_range)
    ax1.legend(fontsize=8)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # -- Subplot 2: C-space (x, psi) projection --
    ax2 = axes[1]
    ax2.set_title("C-space: (x, ψ) projection")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("ψ (rad)")
    if collision_pts:
        cxs = [p[0] for p in collision_pts]
        cpsis = [p[2] for p in collision_pts]
        ax2.scatter(
            cxs, cpsis, s=2, c=obstacle_hex(), alpha=0.3, label="Collision"
        )
    if rrt_path and len(rrt_path) >= 2:
        rxs = [p[0] for p in rrt_path]
        rpsis = [p[2] for p in rrt_path]
        ax2.plot(rxs, rpsis, "b-", linewidth=2, label="RRT*")
    if sst_path and len(sst_path) >= 2:
        sxs = [p[0] for p in sst_path]
        spsis = [p[2] for p in sst_path]
        ax2.plot(sxs, spsis, "g--", linewidth=2, label="SST")
    ax2.plot(
        start_pose[0],
        start_pose[2],
        "s",
        color=annotation_hex(),
        markersize=10,
        label="Start",
    )
    ax2.plot(
        goal_pose[0],
        goal_pose[2],
        "x",
        color=annotation_hex(),
        markersize=12,
        label="Goal",
    )
    ax2.set_xlim(x_range)
    ax2.set_ylim([-math.pi, math.pi])
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # -- Subplot 3: Cartesian 2-D view --
    ax3 = axes[2]
    ax3.set_title("Cartesian 2-D view")
    ax3.set_xlabel("x (m)")
    ax3.set_ylabel("y (m)")
    for obs in obstacles:
        xmin, ymin, xmax, ymax = obs
        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=1,
            edgecolor="black",
            facecolor=obstacle_hex(),
            alpha=0.7,
        )
        ax3.add_patch(rect)
    if rrt_path and len(rrt_path) >= 2:
        rxs = [p[0] for p in rrt_path]
        rys = [p[1] for p in rrt_path]
        ax3.plot(rxs, rys, "b-", linewidth=2, label="RRT*")
    if sst_path and len(sst_path) >= 2:
        sxs = [p[0] for p in sst_path]
        sys_ = [p[1] for p in sst_path]
        ax3.plot(sxs, sys_, "g--", linewidth=2, label="SST")
    ax3.plot(
        start_pose[0],
        start_pose[1],
        "s",
        color=annotation_hex(),
        markersize=10,
        label="Start",
    )
    ax3.plot(
        goal_pose[0],
        goal_pose[1],
        "x",
        color=annotation_hex(),
        markersize=12,
        label="Goal",
    )
    ax3.set_xlim(x_range)
    ax3.set_ylim(y_range)
    ax3.legend(fontsize=8)
    ax3.set_aspect("equal")
    ax3.grid(True, alpha=0.3)

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
