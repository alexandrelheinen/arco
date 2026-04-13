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

import argparse
import logging

from arco.tools.config import load_config
from arco.tools.simulator.main.city import run_race
from arco.tools.simulator.scenes.vehicle import VehicleScene


def main() -> None:
    """Parse CLI arguments and launch the vehicle benchmark race."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fps", type=int, default=30, metavar="N")
    parser.add_argument("--dt", type=float, default=0.1, metavar="S")
    parser.add_argument("--record", metavar="FILE", default="")
    parser.add_argument(
        "--record-duration",
        type=float,
        default=90.0,
        metavar="S",
        dest="record_duration",
    )
    args = parser.parse_args()

    scene = VehicleScene(load_config("vehicle"))
    run_race(
        scene,
        fps=args.fps,
        dt=args.dt,
        record=args.record,
        record_duration=args.record_duration,
    )


if __name__ == "__main__":
    main()
