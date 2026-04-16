"""arcosim CLI: run an ARCO simulation scenario from a YAML file.

Usage::

    arcosim path/to/scenario.yml [--fps N] [--record PATH]
            [--record-duration S]
    arcosim path/to/scenario.yml --image [--record PATH]
    arcosim path/to/scenario.yml --static [--record PATH]

The ``scenario:`` field in the YAML header identifies which simulation to run.
Supported scenarios: astar, city, occ, ppp, rr, rrp, vehicle.

Static / image mode
-------------------
Pass ``--image`` (or its alias ``--static``) to run the example in static
image mode instead of the real-time pygame simulation.  This mode is
equivalent to the former ``arcoex`` command.  When ``--record PATH`` is
also provided, the output image is saved to *PATH* instead of opening an
interactive window.

Examples::

    # interactive simulation
    arcosim map/city.yml

    # static image, open window
    arcosim map/city.yml --image

    # static image, saved to file (replaces: arcoex map/city.yml --save out.png)
    arcosim map/city.yml --image --record out.png

Requires the ``tools`` optional dependency group for static mode::

    pip install arco[tools]

Requires the ``tools`` and ``pygame`` optional dependency groups for
real-time simulation::

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


def _dispatch_static(
    scenario: str,
    cfg: dict[str, Any],
    save_path: str | None,
) -> None:
    """Dispatch to the *static image* example handler for the given scenario.

    This is the mode formerly provided by the ``arcoex`` command.  It imports
    ``tools.examples.<scenario>`` and calls its ``main(cfg, save_path=…)``
    function, producing a matplotlib figure instead of a pygame window.

    Args:
        scenario: Scenario name, e.g. ``"city"`` or ``"ppp"``.
        cfg: Parsed scenario configuration dict.
        save_path: File path to save the output image, or ``None`` to open an
            interactive matplotlib window.

    Raises:
        SystemExit: If the ``tools`` (matplotlib) extra is not installed, or
            if the scenario requires ``pygame`` and that extra is absent.
    """
    import importlib

    try:
        import matplotlib  # noqa: F401  (presence check only)
    except ImportError:
        print(
            "arcosim --image requires the 'tools' extra. "
            "Install with: pip install arco[tools]",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        mod = importlib.import_module(f"arco.tools.examples.{scenario}")
    except ImportError as exc:
        if "pygame" in str(exc) or "OpenGL" in str(exc):
            print(
                f"arcosim --image: scenario '{scenario}' requires the "
                "'pygame' extra. "
                "Install with: pip install arco[tools,pygame]",
                file=sys.stderr,
            )
            sys.exit(1)
        raise

    saved_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        mod.main(cfg, save_path=save_path)
    finally:
        sys.argv = saved_argv


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
        help="Record simulation to this MP4 file (requires ffmpeg). "
        "In --image/--static mode, saves the output image to this path.",
    )
    parser.add_argument(
        "--record-duration",
        type=float,
        default=90.0,
        metavar="SECS",
        dest="record_duration",
        help="Maximum recording length in seconds (default: 90).",
    )
    # --image / --static: static matplotlib rendering (replaces arcoex)
    static_group = parser.add_mutually_exclusive_group()
    static_group.add_argument(
        "--image",
        action="store_true",
        default=False,
        help=(
            "Run in static image mode (matplotlib) instead of real-time "
            "pygame simulation.  Equivalent to the former 'arcoex' command. "
            "Requires: pip install arco[tools]"
        ),
    )
    static_group.add_argument(
        "--static",
        action="store_true",
        default=False,
        dest="static",
        help="Alias for --image.",
    )
    args, extra_argv = parser.parse_known_args()

    scenario, cfg = _load_scenario(args.scenario_file)

    if args.image or args.static:
        # Static image mode — delegates to tools/examples/<scenario>.py
        save_path: str | None = args.record if args.record else None
        _dispatch_static(scenario, cfg, save_path)
    else:
        _dispatch(scenario, cfg, args.record, args.record_duration, extra_argv)


if __name__ == "__main__":
    main()
