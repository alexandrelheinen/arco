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

#: Absolute path to the ``tools/`` directory (two levels above this file).
_TOOLS_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Absolute path to the ``tools/simulator/`` directory (for scenes imports).
_SIMULATOR_DIR: str = os.path.join(_TOOLS_DIR, "simulator")

#: Supported scenario names — must match modules in
#: ``tools/simulator/main/``.
SUPPORTED_SCENARIOS: frozenset[str] = frozenset(
    {"astar", "city", "occ", "ppp", "rr", "rrp", "vehicle"}
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _setup_import_paths() -> None:
    """Prepend ``tools/`` and ``tools/simulator/`` to ``sys.path``.

    This allows the simulator modules (which use relative-style imports such
    as ``import renderer_gl`` or ``from scenes.city import CityScene``) to
    locate their dependencies without needing to be installed as packages.
    """
    for path in (_TOOLS_DIR, _SIMULATOR_DIR):
        if path not in sys.path:
            sys.path.insert(0, path)


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
        mod = importlib.import_module(f"simulator.main.{scenario}")
        mod.main()
    finally:
        sys.argv = saved_argv


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the ``arcosim`` CLI.

    Parses CLI arguments, validates the scenario YAML file, sets up import
    paths, and dispatches to the matching simulator handler.

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

    scenario, _ = _load_scenario(args.scenario_file)
    _setup_import_paths()
    _dispatch(scenario, args.record, args.record_duration, extra_argv)


if __name__ == "__main__":
    main()
