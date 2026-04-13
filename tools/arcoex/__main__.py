"""arcoex CLI: run an ARCO example scenario from a YAML file.

Usage::

    arcoex path/to/scenario.yml [--save PATH]

The ``scenario:`` field in the YAML header identifies which example to run.
Supported scenarios: astar, city, occ, ppp, rr, rrp, vehicle.

Requires the ``tools`` optional dependency group::

    pip install arco[tools]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

# ---------------------------------------------------------------------------
# Optional-dependency guard — checked before any further imports.
# ---------------------------------------------------------------------------
try:
    import yaml
    import matplotlib  # noqa: F401  (presence check only)
except ImportError:
    print(
        "arcoex requires the 'tools' extra. "
        "Install with: pip install arco[tools]",
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

#: Supported scenario names — must match modules in ``tools/examples/``.
SUPPORTED_SCENARIOS: frozenset[str] = frozenset(
    {"astar", "city", "occ", "ppp", "rr", "rrp", "vehicle"}
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _setup_import_paths() -> None:
    """Prepend ``tools/`` and ``tools/simulator/`` to ``sys.path``.

    This allows the example modules (which use relative-style imports such as
    ``from config import load_config``) to locate their dependencies without
    needing to be installed as proper packages.
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
        print(f"arcoex: scenario file not found: {path!r}", file=sys.stderr)
        sys.exit(1)
    with open(path) as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh) or {}
    scenario = cfg.get("scenario")
    if not scenario:
        print(
            "arcoex: the YAML file must declare 'scenario:' as its first key.",
            file=sys.stderr,
        )
        sys.exit(1)
    if scenario not in SUPPORTED_SCENARIOS:
        print(
            f"arcoex: unknown scenario {scenario!r}. "
            f"Supported values: {sorted(SUPPORTED_SCENARIOS)}",
            file=sys.stderr,
        )
        sys.exit(1)
    return scenario, cfg


def _dispatch(scenario: str, save_path: str | None) -> None:
    """Dispatch to the example handler for the given scenario.

    Imports ``tools.examples.<scenario>`` dynamically (after path setup) and
    calls its ``main(save_path=...)`` function.

    Args:
        scenario: Scenario name, e.g. ``"city"`` or ``"ppp"``.
        save_path: Optional file path to save the output image to.  When
            ``None`` the example opens an interactive matplotlib window.

    Raises:
        SystemExit: If optional pygame dependencies are missing for the
            requested scenario.
    """
    import importlib

    try:
        mod = importlib.import_module(f"examples.{scenario}")
    except ImportError as exc:
        if "pygame" in str(exc) or "OpenGL" in str(exc):
            print(
                f"arcoex: scenario '{scenario}' requires the 'pygame' extra."
                " Install with: pip install arco[tools,pygame]",
                file=sys.stderr,
            )
            sys.exit(1)
        raise
    mod.main(save_path=save_path)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the ``arcoex`` CLI.

    Parses CLI arguments, validates the scenario YAML file, sets up import
    paths, and dispatches to the matching example handler.

    Raises:
        SystemExit: On any validation error or missing dependencies.
    """
    parser = argparse.ArgumentParser(
        prog="arcoex",
        description=(
            "Run an ARCO example scenario from a YAML file.\n\n"
            f"Supported scenarios: {', '.join(sorted(SUPPORTED_SCENARIOS))}"
        ),
    )
    parser.add_argument(
        "scenario_file",
        metavar="SCENARIO",
        help="Path to the scenario .yml file (must contain 'scenario:' key).",
    )
    parser.add_argument(
        "--save",
        metavar="PATH",
        default=None,
        help="Save the output image to PATH instead of opening a window.",
    )
    args = parser.parse_args()

    scenario, _ = _load_scenario(args.scenario_file)
    _setup_import_paths()
    _dispatch(scenario, args.save)


if __name__ == "__main__":
    main()
