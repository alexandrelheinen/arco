#!/usr/bin/env python
"""Vehicle benchmark simulator - RRT* vs SST on a shared 2-D map.

Usage
-----
::

    cd tools/simulator
    python main/vehicle.py

Optional flags::

    python main/vehicle.py --fps 30
    python main/vehicle.py --record /tmp/vehicle.mp4 --record-duration 90
"""

from __future__ import annotations

import logging

from arco.config import load_config
from arco.simulator.main.city import run_race
from arco.simulator.scenes.vehicle import VehicleScene


def main(cfg: dict, save_path: str | None, sim_duration: float) -> None:
    """Parse CLI arguments and launch the vehicle benchmark race."""
    sim_cfg = load_config("simulator")
    scene = VehicleScene(cfg)
    run_race(
        scene,
        fps=sim_cfg["fps"],
        dt=sim_cfg["timestep"],
        record=save_path,
        record_duration=sim_duration,
    )
