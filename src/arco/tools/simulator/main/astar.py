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

import argparse
import logging

from arco.tools.simulator.scenes.astar import AStarScene
from arco.tools.simulator.sim import run_sim


def main(cfg: dict) -> None:
    """Parse CLI arguments and launch the A* simulator."""
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
        help="Fit initial view to the planned route bounding box.",
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

    scene = AStarScene(
        cfg.get("graph", {}),
        cfg.get("vehicle", {}),
    )
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
    import argparse as _argparse

    import yaml as _yaml

    _parser = _argparse.ArgumentParser()
    _parser.add_argument("scenario", metavar="FILE")
    _args, _rest = _parser.parse_known_args()
    with open(_args.scenario) as _fh:
        _cfg = _yaml.safe_load(_fh) or {}
    import sys as _sys

    _sys.argv = [_sys.argv[0], *_rest]
    main(_cfg)
