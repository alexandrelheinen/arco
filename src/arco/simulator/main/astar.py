#!/usr/bin/env python
"""A* route-planning simulator — thin entrypoint.

Builds a procedural road network, plans an A* route, and runs the unified
two-phase ARCO simulator loop (background backdrop + vehicle tracking).

Keyboard controls
-----------------
SPACE         Pause / resume simulation
R             Restart from the beginning
C             Toggle camera mode (full view / follow vehicle)
\\+ / -        Zoom in / out in follow-vehicle camera mode
Q / Escape    Quit

Usage
-----
::

    cd tools/simulator
    python main/astar.py

Optional flags::

    python main/astar.py --fps 60
    python main/astar.py --dt 0.05
    python main/astar.py --camera follow
    python main/astar.py --zoom
    python main/astar.py --record out.mp4 --record-duration 45
"""

from __future__ import annotations

import logging

from arco.config import load_config
from arco.simulator.scenes.astar import AStarScene
from arco.simulator.sim import run_sim


def main(cfg: dict, save_path: str | None, sim_duration: float) -> None:
    sim_cfg = load_config("simulator")
    scene = AStarScene(
        cfg.get("graph", {}),
        cfg.get("vehicle", {}),
    )
    run_sim(
        scene,
        fps=sim_cfg["fps"],
        dt=sim_cfg["timestep"],
        camera="full",
        zoom=0.5,
        record=save_path,
        record_duration=sim_duration,
    )
