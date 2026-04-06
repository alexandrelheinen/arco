#!/usr/bin/env python
"""SST planning simulator — thin entrypoint.

Runs the SST algorithm to completion on a 2-D obstacle environment, reveals
the exploration tree incrementally, then tracks the solution path with a
Dubins vehicle.

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
    python main/sst.py

Optional flags::

    python main/sst.py --fps 30
    python main/sst.py --zoom
    python main/sst.py --close
    python main/sst.py --record /tmp/sst.mp4 --record-duration 60
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Make arco and tools packages importable without a full install.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_HERE, ".."))

from scenes.sst import SSTScene
from sim import run_sim

from config import load_config


def main() -> None:
    """Parse CLI arguments and launch the SST simulator."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        metavar="N",
        help="Target frame rate (default: 30).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        metavar="S",
        help="Simulation timestep in seconds (default: 0.1).",
    )
    parser.add_argument(
        "--camera",
        choices=["full", "follow"],
        default="full",
        help="Starting camera mode (default: full).",
    )
    parser.add_argument(
        "--zoom",
        action="store_true",
        default=False,
        help="Fit initial view to the path bounding box.",
    )
    parser.add_argument(
        "--record",
        metavar="FILE",
        default="",
        help="Record a headless MP4 to FILE (requires ffmpeg).",
    )
    parser.add_argument(
        "--record-duration",
        type=float,
        default=60.0,
        metavar="S",
        dest="record_duration",
        help="Maximum recording duration in seconds (default: 60).",
    )
    parser.add_argument(
        "--close",
        action="store_true",
        default=False,
        help="Close the window automatically when the vehicle reaches the goal.",
    )
    args = parser.parse_args()

    scene = SSTScene(load_config("sst"))
    run_sim(
        scene,
        fps=args.fps,
        dt=args.dt,
        camera=args.camera,
        zoom=args.zoom,
        record=args.record,
        record_duration=args.record_duration,
        auto_close=args.close,
    )


if __name__ == "__main__":
    main()
