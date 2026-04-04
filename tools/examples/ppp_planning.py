"""PPP (triple prismatic) robot planning in a 3-D warehouse environment.

Runs RRT* and SST planners side-by-side on an industrial warehouse bay
modelled as a box-shaped work volume (20 m × 10 m × 6 m) with ground-level
box obstacles.  A full-width blocking wall at x = 7–9 forces both planners
to arc over it (z > 4.5 m).

Only the chosen path is rendered — no exploration tree — to keep the 3-D
view uncluttered.

Usage
-----
Run interactively (opens a matplotlib window)::

    python tools/examples/ppp_planning.py

Save the output image without opening a window::

    python tools/examples/ppp_planning.py --save path/to/output.png
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulator"))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from logging_config import configure_logging
from scenes.ppp import BOXES as _BOXES
from scenes.ppp import (
    _sample_box_surface,
)
from scenes.ppp import is_wall as _is_wall

from arco.mapping import KDTreeOccupancy
from arco.planning.continuous import RRTPlanner, SSTPlanner
from config import load_config

logger = logging.getLogger(__name__)

_cfg = load_config("ppp")

_START = np.array([1.0, 1.0, 0.0])
_GOAL = np.array([19.0, 9.0, 0.0])


# ---------------------------------------------------------------------------
# Occupancy map
# ---------------------------------------------------------------------------


def build_occupancy() -> KDTreeOccupancy:
    """Build the 3-D warehouse obstacle map.

    Samples the surface of every box (using the shared
    :func:`~scenes.ppp._sample_box_surface` helper) and wraps the combined
    point cloud in a :class:`~arco.mapping.KDTreeOccupancy`.

    Returns:
        A collision-query-ready 3-D occupancy map.
    """
    all_pts: list[list[float]] = []
    for box in _BOXES:
        all_pts.extend(_sample_box_surface(*box))
    return KDTreeOccupancy(
        all_pts, clearance=float(_cfg["obstacle_clearance"])
    )


# ---------------------------------------------------------------------------
# 3-D box renderer (matplotlib)
# ---------------------------------------------------------------------------


def _draw_box(
    ax: plt.Axes,
    x1: float,
    y1: float,
    z1: float,
    x2: float,
    y2: float,
    z2: float,
    color: str = "saddlebrown",
    alpha: float = 0.40,
) -> None:
    """Render an axis-aligned box with ``bar3d``.

    Args:
        ax: Matplotlib 3-D axes.
        x1: Minimum x.
        y1: Minimum y.
        z1: Minimum z.
        x2: Maximum x.
        y2: Maximum y.
        z2: Maximum z.
        color: Face colour string.
        alpha: Face transparency in [0, 1].
    """
    ax.bar3d(  # type: ignore[attr-defined]
        x1,
        y1,
        z1,
        x2 - x1,
        y2 - y1,
        z2 - z1,
        color=color,
        alpha=alpha,
        shade=True,
        zsort="average",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(save_path: str | None = None) -> None:
    """Run RRT* and SST on the 3-D PPP warehouse and display the result.

    Args:
        save_path: If provided, save the figure to this path instead of
            opening an interactive window.  The parent directory is created
            automatically if it does not exist.
    """
    if save_path is not None:
        matplotlib.use("Agg")

    bounds = [tuple(b) for b in _cfg["bounds"]]
    occ = build_occupancy()

    # --- RRT* ---------------------------------------------------------------
    rrt = RRTPlanner(
        occ,
        bounds=bounds,
        max_sample_count=int(_cfg["rrt_max_sample_count"]),
        step_size=float(_cfg["step_size"]),
        goal_tolerance=float(_cfg["goal_tolerance"]),
        collision_check_count=int(_cfg["collision_check_count"]),
        goal_bias=float(_cfg["goal_bias"]),
        early_stop=True,
    )
    logger.info("Running RRT* in 3-D …")
    _, _, rrt_path = rrt.get_tree(_START.copy(), _GOAL.copy())
    rrt_len = 0.0
    if rrt_path is not None:
        rrt_len = sum(
            float(np.linalg.norm(rrt_path[i + 1] - rrt_path[i]))
            for i in range(len(rrt_path) - 1)
        )
        logger.info(
            "RRT*: %d waypoints, length=%.1f m", len(rrt_path), rrt_len
        )
    else:
        logger.warning("RRT*: no path found.")

    # --- SST ----------------------------------------------------------------
    sst = SSTPlanner(
        occ,
        bounds=bounds,
        max_sample_count=int(_cfg["sst_max_sample_count"]),
        step_size=float(_cfg["step_size"]),
        goal_tolerance=float(_cfg["goal_tolerance"]),
        witness_radius=float(_cfg["witness_radius"]),
        collision_check_count=int(_cfg["collision_check_count"]),
        goal_bias=float(_cfg["goal_bias"]),
        early_stop=True,
    )
    logger.info("Running SST in 3-D …")
    _, _, sst_path = sst.get_tree(_START.copy(), _GOAL.copy())
    sst_len = 0.0
    if sst_path is not None:
        sst_len = sum(
            float(np.linalg.norm(sst_path[i + 1] - sst_path[i]))
            for i in range(len(sst_path) - 1)
        )
        logger.info("SST: %d waypoints, length=%.1f m", len(sst_path), sst_len)
    else:
        logger.warning("SST: no path found.")

    # --- 3-D figure ---------------------------------------------------------
    fig = plt.figure(figsize=(14, 7))

    specs = [
        ("RRT* — 3-D PPP warehouse", rrt_path, rrt_len, "royalblue"),
        ("SST — 3-D PPP warehouse", sst_path, sst_len, "mediumseagreen"),
    ]
    x_lim = (
        float(_cfg["bounds"][0][0]),
        float(_cfg["bounds"][0][1]),
    )
    y_lim = (
        float(_cfg["bounds"][1][0]),
        float(_cfg["bounds"][1][1]),
    )
    z_lim = (0.0, float(_cfg["bounds"][2][1]))

    for col, (title, path, length, color) in enumerate(specs):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")

        # Obstacle boxes — wall is coloured distinctly from scatter boxes.
        for box in _BOXES:
            _draw_box(
                ax,
                *box,
                color="sienna" if _is_wall(box) else "peru",
                alpha=0.45,
            )

        # Solution path
        if path is not None and len(path) >= 2:
            arr = np.array(path)
            label = f"Path  {length:.1f} m | {len(path)} wpts"
            ax.plot(  # type: ignore[attr-defined]
                arr[:, 0],
                arr[:, 1],
                arr[:, 2],
                color=color,
                linewidth=2.5,
                zorder=5,
                label=label,
            )

        # Start and goal markers
        ax.scatter(  # type: ignore[attr-defined]
            [_START[0]],
            [_START[1]],
            [_START[2]],
            color="limegreen",
            s=80,
            zorder=6,
            label="Start",
        )
        ax.scatter(  # type: ignore[attr-defined]
            [_GOAL[0]],
            [_GOAL[1]],
            [_GOAL[2]],
            color="orangered",
            marker="*",
            s=120,
            zorder=6,
            label="Goal",
        )

        ax.set_xlim(*x_lim)  # type: ignore[attr-defined]
        ax.set_ylim(*y_lim)  # type: ignore[attr-defined]
        ax.set_zlim(*z_lim)  # type: ignore[attr-defined]
        ax.set_xlabel("X (m)")  # type: ignore[attr-defined]
        ax.set_ylabel("Y (m)")  # type: ignore[attr-defined]
        ax.set_zlabel("Z (m)")  # type: ignore[attr-defined]
        n_wpts = len(path) if path else 0
        subtitle = f"Length {length:.1f} m | {n_wpts} waypoints"
        ax.set_title(f"{title}\n{subtitle}")  # type: ignore[attr-defined]
        ax.legend(loc="upper left", fontsize=8)  # type: ignore[attr-defined]
        ax.view_init(elev=25, azim=-50)  # type: ignore[attr-defined]

    plt.suptitle(
        "PPP robot — RRT* vs SST in 3-D warehouse",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info("Saved PPP planning example to %s", save_path)
    else:
        plt.show()


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save",
        metavar="PATH",
        default=None,
        help="Save the figure instead of opening an interactive window.",
    )
    args = parser.parse_args()
    main(save_path=args.save)
