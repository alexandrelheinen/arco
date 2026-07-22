"""Headless city A* sharp-corner / MPC-horizon diagnostic figure.

Builds the seeded ``map/city.yml`` scene, plots the optimized A* reference
with heading-change markers, and annotates the old full-block horizon
(~108 m) versus the half-block city default (~64.8 m).

Output::

    tools/output/city_astar_mpc_horizon_demo.png

Usage::

    python tools/city_astar_mpc_horizon_demo.py
"""

from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

# CityScene imports renderer_gl at module load; stub it for headless runs.
_stub = types.ModuleType("arco.simulator.renderer_gl")
for _name in (
    "draw_disk",
    "draw_ring",
    "draw_polyline",
    "draw_path",
    "draw_rect",
    "draw_triangle",
    "draw_text",
    "draw_obstacle_points",
    "draw_tree",
):
    setattr(_stub, _name, lambda *a, **k: None)
sys.modules["arco.simulator.renderer_gl"] = _stub

from arco.simulator.scenes.sparse import CityScene  # noqa: E402
from arco.simulator.sim.city_race_style import (  # noqa: E402
    DEFAULT_CITY_HORIZON_DT,
    DEFAULT_CITY_HORIZON_STEP_COUNT,
)


def _heading_changes(
    pts: np.ndarray, min_deg: float = 45.0
) -> list[tuple[int, float, float]]:
    """Return ``(index, arc_s, delta_heading_deg)`` for sharp corners."""
    if len(pts) < 3:
        return []
    cum = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    )
    out: list[tuple[int, float, float]] = []
    for i in range(1, len(pts) - 1):
        v1 = pts[i] - pts[i - 1]
        v2 = pts[i + 1] - pts[i]
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        a1 = math.atan2(float(v1[1]), float(v1[0]))
        a2 = math.atan2(float(v2[1]), float(v2[0]))
        dang = math.degrees(math.atan2(math.sin(a2 - a1), math.cos(a2 - a1)))
        if abs(dang) >= min_deg:
            out.append((i, float(cum[i]), dang))
    return out


def main() -> Path:
    """Render the diagnostic figure and return its path."""
    root = Path(__file__).resolve().parents[1]
    with open(root / "map" / "city.yml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    scene = CityScene(
        cfg.get("planner", {}),
        cfg.get("world", {}),
        sim_cfg=cfg.get("simulator", {}),
    )
    scene.build()
    pts = np.asarray(scene.astar_waypoints, dtype=float)
    cruise = float(scene.vehicle_config.cruise_speed)
    h_new = cruise * DEFAULT_CITY_HORIZON_DT * DEFAULT_CITY_HORIZON_STEP_COUNT
    h_old = cruise * 0.05 * 120

    corners = _heading_changes(pts)
    fig, ax = plt.subplots(figsize=(8.5, 8.0), dpi=140)
    ax.set_facecolor("#101418")
    fig.patch.set_facecolor("#101418")
    ax.plot(
        pts[:, 0], pts[:, 1], color="#b388ff", lw=1.8, label="A* reference"
    )
    ax.scatter(
        [pts[0, 0], pts[-1, 0]],
        [pts[0, 1], pts[-1, 1]],
        c=["#ffffff", "#ffd54f"],
        s=40,
        zorder=3,
        label="start / goal",
    )
    if corners:
        cxy = pts[[i for i, _, _ in corners]]
        ax.scatter(
            cxy[:, 0],
            cxy[:, 1],
            c="#ff8a65",
            s=36,
            zorder=4,
            label=f"turns ≥45° ({len(corners)})",
        )
        # Annotate the late sharp pair that trapped the release racer.
        late = [c for c in corners if c[1] > 700.0]
        if late:
            i, s, dang = late[0]
            ax.annotate(
                f"s≈{s:.0f} m\nΔψ={dang:.0f}°",
                xy=(pts[i, 0], pts[i, 1]),
                xytext=(pts[i, 0] - 90, pts[i, 1] - 40),
                color="#ffccbc",
                fontsize=8,
                arrowprops={"arrowstyle": "->", "color": "#ffccbc"},
            )

    # Horizon scale bars near the mid-path point.
    mid = len(pts) // 2
    origin = pts[mid]
    ax.plot(
        [origin[0], origin[0] + h_old],
        [origin[1] + 18, origin[1] + 18],
        color="#90caf9",
        lw=3.0,
        solid_capstyle="butt",
        label=f"old horizon {h_old:.0f} m (1 block)",
    )
    ax.plot(
        [origin[0], origin[0] + h_new],
        [origin[1] + 8, origin[1] + 8],
        color="#80cbc4",
        lw=3.0,
        solid_capstyle="butt",
        label=f"new horizon {h_new:.0f} m (½ block)",
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        "City A* reference: sharp grid corners vs MPC horizon",
        color="white",
        fontsize=11,
    )
    ax.tick_params(colors="#b0bec5")
    for spine in ax.spines.values():
        spine.set_color("#455a64")
    ax.legend(
        loc="lower right",
        fontsize=8,
        facecolor="#1b2228",
        edgecolor="#455a64",
        labelcolor="white",
    )
    ax.set_xlabel("x [m]", color="#b0bec5")
    ax.set_ylabel("y [m]", color="#b0bec5")

    out_dir = root / "tools" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "city_astar_mpc_horizon_demo.png"
    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


if __name__ == "__main__":
    main()
