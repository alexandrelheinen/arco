"""arcosim CLI: run an ARCO simulation scenario from a YAML file.

Usage::

    arcosim path/to/scenario.yml [--record PATH] [--record-duration SECONDS]]
    arcosim path/to/scenario.yml --static [--record PATH]

Requires the ``tools`` optional dependency group::

    pip install arco[tools]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any

import arco.simulator.main as simulator

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


def _dispatch(
    cfg: dict[str, Any], save_path: str | None, record_duration: float
) -> None:
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
        help="Destination of the output file. If not provided, display interactively (default: none).",
    )
    parser.add_argument(
        "--record-duration",
        "-d",
        type=float,
        default=360.0,
        help="Maximum recording length in seconds (default: 360 s).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the ``arcosim`` CLI.

    Parses CLI arguments, validates the scenario YAML file, and dispatches
    to the matching simulator handler.

    Unknown flags are forwarded verbatim to the underlying simulator so that
    scenario-specific options (e.g. ``--fps``, ``--dt``, ``--camera``) can be
    passed through without needing to be declared here.

    Raises:
        SystemExit: On any validation error or missing dependencies.
    """
    # Delegate to the appropriate handler
    args = parse_args()
    cfg = _load_map(args.scenario_file)
    _dispatch(cfg, args.output, args.record_duration)


if __name__ == "__main__":
    main()
