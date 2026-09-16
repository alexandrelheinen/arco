"""arcosim CLI: run an ARCO simulation scenario from a YAML file.

Usage::

    arcosim path/to/scenario.yml [-o PATH] [-d SECONDS] [--fast-record]
    arcosim path/to/scenario.yml -o PATH.png --still FRAME[,FRAME...]

Requires the ``tools`` optional dependency group::

    pip install arco[tools]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

from arco.simulator.sim import still

logger = logging.getLogger(__name__)

# Optional-dependency guard for yaml
try:
    import yaml

except ImportError:
    print(
        "arcosim requires the 'tools' extra. "
        "Install with: pip install arco[tools]",
        file=sys.stderr,
    )
    sys.exit(1)


def _load_map(path: str) -> dict[str, Any]:
    """Load a map YAML file and return the scenario configuration.

    Args:
        path: File-system path to the map ``.yml`` file.

    Returns:
        A dictionary containing the scenario configuration.

    Raises:
        SystemExit: If the file is not found, the ``scenario:`` key is
            missing, or the scenario name is not supported.
    """
    logger.info("Loading map %r...", path)

    if not os.path.isfile(path):
        logger.error("Map file not found: %r", path)
        sys.exit(1)
    with open(path) as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh) or {}

    # Only a single sanity check here is enough
    scenario = cfg.get("scenario")
    if not scenario:
        logger.error("Map file %r is missing the 'scenario:' key.", path)
        sys.exit(1)

    return cfg


def _apply_fast_record(cfg: dict[str, Any]) -> None:
    """Set ``simulator.fast_record`` on a loaded scenario config.

    Args:
        cfg: Mutable scenario dictionary (must already contain ``scenario``).
    """
    # Scenes read simulator.fast_record to skip tree-reveal pacing in
    # headless release / CI recordings (see docs/VISUALIZATION.md).
    sim = cfg.setdefault("simulator", {})
    if not isinstance(sim, dict):
        cfg["simulator"] = {"fast_record": True}
    else:
        sim["fast_record"] = True


def _dispatch(
    cfg: dict[str, Any],
    save_path: str | None,
    record_duration: float,
    *,
    fast_record: bool = False,
) -> None:
    if fast_record:
        _apply_fast_record(cfg)
    # Lazy import: scenario mains pull pygame/OpenGL.  Keeping this inside
    # dispatch lets unit tests import parse_args / _apply_fast_record without
    # the display stack (headless CI unit jobs omit pygame).
    import arco.simulator.main as simulator

    scenario = cfg["scenario"]
    submodule = getattr(simulator, scenario, None)
    if not submodule:
        logger.error(
            "Unsupported scenario type %r",
            scenario,
        )
        sys.exit(1)

    handler = getattr(submodule, "main", None)
    if not handler:
        logger.error(
            "Scenario %r does not have a 'main' handler function.",
            scenario,
        )
        sys.exit(1)

    handler(cfg, save_path, record_duration)


def _build_still_request(
    still_spec: str,
    output: str | None,
    width: int | None,
    height: int | None,
) -> still.StillRequest:
    """Turn the still-related CLI arguments into a request.

    Args:
        still_spec: Comma-separated frame indices from ``--still``.
        output: Destination path from ``--output``.
        width: Framebuffer width from ``--width``, or ``None``.
        height: Framebuffer height from ``--height``, or ``None``.

    Returns:
        The :class:`~arco.simulator.sim.still.StillRequest` to install.

    Raises:
        SystemExit: If no output path was given or the frame list is
            unparsable.
    """
    if not output:
        logger.error("--still needs an output path: pass -o FILE.png")
        sys.exit(1)
    try:
        frames = still.parse_frames(still_spec)
    except ValueError as exc:
        logger.error("Invalid --still value %r: %s", still_spec, exc)
        sys.exit(1)
    return still.StillRequest(
        frames=frames,
        output=Path(output),
        width=width,
        height=height,
    )


def _still_duration(
    last_frame: int, record_duration: float, fps: int
) -> float:
    """Return a recording budget that reaches *last_frame*.

    Args:
        last_frame: Highest requested frame index.
        record_duration: Budget requested on the command line.
        fps: Scenario frame rate in frames per second.

    Returns:
        The larger of *record_duration* and the time needed to reach
        *last_frame*, plus a one-second margin.
    """
    needed = (last_frame + 2) / float(max(fps, 1)) + 1.0
    return max(record_duration, needed)


def _report_missing_frames() -> None:
    """Fail when the run ended before every requested frame.

    Raises:
        SystemExit: If the still sink never reached some requested frames.
    """
    sink = still.last_sink()
    if sink is None:
        return
    missing = sink.missing_frames()
    if missing:
        logger.error(
            "Scenario ended before frame(s) %s; pick a lower index.",
            list(missing),
        )
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the ``arcosim`` command.

    Returns:
        An argparse.Namespace containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="arcosim",
        description=("Run an ARCO simulation scenario from a YAML map file."),
    )
    parser.add_argument(
        "scenario_file",
        help="Path to the scenario YAML file (must contain 'scenario:' key).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help=(
            "Destination of the output file. If not provided, display "
            "interactively (default: none)."
        ),
    )
    parser.add_argument(
        "--record-duration",
        "-d",
        type=float,
        default=360.0,
        help="Maximum recording length in seconds (default: 360 s).",
    )
    parser.add_argument(
        "--still",
        default=None,
        metavar="FRAMES",
        help=(
            "Save recorded frames as PNG instead of an MP4.  Takes one "
            "frame index or a comma-separated list; --output names the "
            "file (several frames add an _fNNNNN suffix)."
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Still framebuffer width (default: the scenario's own).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Still framebuffer height (default: the scenario's own).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Pin unseeded planner sampling so a run repeats exactly.  "
            "Seeds already set in the scenario YAML are left alone."
        ),
    )
    parser.add_argument(
        "--fast-record",
        action="store_true",
        help=(
            "Headless release mode: skip animated planner-tree reveal and "
            "spend the recording budget on the race / tracking phase.  "
            "Sets simulator.fast_record in the loaded YAML config."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the ``arcosim`` CLI.

    Parses CLI arguments, validates the scenario YAML file, and dispatches
    to the matching simulator handler.

    Raises:
        SystemExit: On any validation error or missing dependencies.
    """
    args = parse_args()
    cfg = _load_map(args.scenario_file)

    if args.seed is not None:
        still.pin_unseeded_rng(int(args.seed))

    record_duration = args.record_duration
    if args.still:
        # Still capture writes one file per frame and is otherwise silent;
        # surface the progress logs so the user sees what was saved.
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        request = _build_still_request(
            args.still, args.output, args.width, args.height
        )
        still.set_request(request)
        from arco.config import load_config  # noqa: PLC0415

        fps = int(load_config("simulator")["fps"])
        record_duration = _still_duration(
            request.last_frame, record_duration, fps
        )

    try:
        _dispatch(
            cfg,
            args.output,
            record_duration,
            fast_record=bool(args.fast_record),
        )
    except still.CaptureDone:
        # Every requested frame was saved; the sink unwound the render loop.
        pass

    if args.still:
        _report_missing_frames()


if __name__ == "__main__":
    main()
