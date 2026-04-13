"""arcosim CLI: run an ARCO simulation scenario from a YAML file.

Usage::

    arcosim path/to/scenario.yml [--fps N] [--record PATH]
            [--record-duration S]

The ``scenario:`` field in the YAML header identifies which simulation to run.
Supported scenarios: astar, city, occ, ppp, rr, rrp, vehicle.

Requires the ``tools`` and ``pygame`` optional dependency groups::

    pip install arco[tools,pygame]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

# ---------------------------------------------------------------------------
# Optional-dependency guard for yaml — needed for _load_scenario at import time.
# ---------------------------------------------------------------------------
try:
    import yaml
except ImportError:
    print(
        "arcosim requires the 'tools' extra. "
        "Install with: pip install arco[tools,pygame]",
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Supported scenario names — must match modules in
#: ``tools/simulator/main/``.
SUPPORTED_SCENARIOS: frozenset[str] = frozenset(
    {"astar", "city", "occ", "ppp", "rr", "rrp", "vehicle"}
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_scenario(path: str) -> tuple[str, dict[str, Any]]:
    """Load a scenario YAML file and return the scenario name and config.

    Args:
        path: File-system path to the scenario ``.yml`` file.

    Returns:
        A ``(scenario_name, config_dict)`` tuple.

    Raises:
        SystemExit: If the file is not found, the ``scenario:`` key is
            missing, or the scenario name is not in
            :data:`SUPPORTED_SCENARIOS`.
    """
    print(f"arcosim: loading scenario {path!r}...")

    if not os.path.isfile(path):
        print(f"arcosim: scenario file not found: {path!r}", file=sys.stderr)
        sys.exit(1)
    with open(path) as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh) or {}

    scenario = cfg.get("scenario")
    if not scenario:
        print(
            "arcosim: the YAML file must declare 'scenario:' as its first"
            " key.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"arcosim: scenario declared: {scenario!r}")

    if scenario not in SUPPORTED_SCENARIOS:
        print(
            f"arcosim: unknown scenario {scenario!r}. "
            f"Supported values: {sorted(SUPPORTED_SCENARIOS)}",
            file=sys.stderr,
        )
        sys.exit(1)
    return scenario, cfg


def _dispatch(
    scenario: str,
    cfg: dict[str, Any],
    record: str,
    record_duration: float,
    extra_argv: list[str],
) -> None:
    """Dispatch to the simulator handler for the given scenario.

    Temporarily replaces ``sys.argv`` so the simulator's own ``argparse``
    call sees the correct flags, then imports and calls
    ``tools.simulator.main.<scenario>.main()``.

    Only ``--record`` and ``--record-duration`` are forwarded automatically
    because they are accepted by every simulator.  Scenario-specific flags
    (e.g. ``--fps``, ``--dt``, ``--camera``) should be passed via
    ``extra_argv`` so they reach the underlying argparser unmodified.

    Args:
        scenario: Scenario name, e.g. ``"city"`` or ``"ppp"``.
        cfg: Parsed scenario configuration dict (already loaded by the CLI).
        record: Output MP4 file path, or an empty string to run interactively.
        record_duration: Maximum recording duration in seconds.
        extra_argv: Additional flags forwarded verbatim to the simulator
            module (e.g. ``["--fps", "60", "--camera", "follow"]``).

    Raises:
        SystemExit: If the ``pygame`` / ``PyOpenGL`` extras are not installed.
    """
    import importlib

    try:
        import OpenGL  # noqa: F401  (presence check only)
        import pygame  # noqa: F401  (presence check only)
    except ImportError:
        print(
            "arcosim requires the 'pygame' extra. "
            "Install with: pip install arco[tools,pygame]",
            file=sys.stderr,
        )
        sys.exit(1)

    sim_argv: list[str] = [f"arcosim_{scenario}"]
    if record:
        sim_argv += [
            "--record",
            record,
            "--record-duration",
            str(record_duration),
        ]
    sim_argv += extra_argv

    saved_argv = sys.argv
    sys.argv = sim_argv
    try:
        mod = importlib.import_module(f"arco.tools.simulator.main.{scenario}")
        mod.main(cfg)
    finally:
        sys.argv = saved_argv


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


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
    parser = argparse.ArgumentParser(
        prog="arcosim",
        description=(
            "Run an ARCO simulation scenario from a YAML file.\n\n"
            f"Supported scenarios: {', '.join(sorted(SUPPORTED_SCENARIOS))}"
        ),
    )
    parser.add_argument(
        "scenario_file",
        metavar="SCENARIO",
        help="Path to the scenario .yml file (must contain 'scenario:' key).",
    )
    parser.add_argument(
        "--record",
        metavar="PATH",
        default="",
        help="Record simulation to this MP4 file (requires ffmpeg).",
    )
    parser.add_argument(
        "--record-duration",
        type=float,
        default=90.0,
        metavar="SECS",
        dest="record_duration",
        help="Maximum recording length in seconds (default: 90).",
    )
    args, extra_argv = parser.parse_known_args()

    scenario, cfg = _load_scenario(args.scenario_file)
    _dispatch(scenario, cfg, args.record, args.record_duration, extra_argv)


if __name__ == "__main__":
    main()
