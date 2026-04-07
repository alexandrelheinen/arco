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
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_HERE, ".."))

from city import run_race
from scenes.vehicle import VehicleScene

from config import load_config


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
