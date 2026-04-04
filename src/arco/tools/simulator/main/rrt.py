#!/usr/bin/env python
"""RRT* planning simulator — thin entrypoint.

Runs the RRT* algorithm to completion on a 2-D obstacle environment, reveals
the exploration tree incrementally, then tracks the solution path with a
Dubins vehicle.

Keyboard controls
-----------------
SPACE         Pause / resume simulation
R             Restart from the beginning
C             Toggle camera mode (full view / follow vehicle)
\+ / -        Zoom in / out in follow-vehicle camera mode
Q / Escape    Quit

Usage
-----
::

    python -m arco.tools.simulator.main.rrt

Optional flags::

    python -m arco.tools.simulator.main.rrt --fps 30
    python -m arco.tools.simulator.main.rrt --zoom
    python -m arco.tools.simulator.main.rrt --record /tmp/rrt.mp4 --record-duration 60
"""

from __future__ import annotations

import argparse
import logging
import os

# Make arco and tools packages importable without a full install.
_HERE = os.path.dirname(os.path.abspath(__file__))

from arco.tools.config import load_config
from arco.tools.simulator.scenes.rrt import RRTScene
from arco.tools.simulator.sim import run_sim


def main() -> None:
    """Parse CLI arguments and launch the RRT* simulator."""
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
    args = parser.parse_args()

    scene = RRTScene(load_config("rrt"))
    run_sim(
        scene,
        fps=args.fps,
        dt=args.dt,
        camera=args.camera,
        zoom=args.zoom,
        record=args.record,
        record_duration=args.record_duration,
    )


if __name__ == "__main__":
    main()
