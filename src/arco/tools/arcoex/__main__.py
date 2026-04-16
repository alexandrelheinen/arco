"""arcoex CLI (deprecated) — thin wrapper over ``arcosim --image``.

.. deprecated::
    ``arcoex`` is deprecated and will be removed in a future release.
    Use ``arcosim <scenario.yml> --image [--record <output.png>]`` instead::

        # was: arcoex map/city.yml --save output.png
        arcosim map/city.yml --image --record output.png

The supported scenarios, YAML format, and output are identical.

Requires the ``tools`` optional dependency group::

    pip install arco[tools]
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Any

# ---------------------------------------------------------------------------
# Optional-dependency guard — checked before any further imports.
# ---------------------------------------------------------------------------
try:
    import matplotlib  # noqa: F401  (presence check only)
    import yaml
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

#: Supported scenario names — must match modules in ``tools/examples/``.
SUPPORTED_SCENARIOS: frozenset[str] = frozenset(
    {"astar", "city", "occ", "ppp", "rr", "rrp", "vehicle"}
)

# ---------------------------------------------------------------------------
# Deprecation warning text
# ---------------------------------------------------------------------------

_DEPRECATION_MSG = (
    "arcoex is deprecated and will be removed in a future release. "
    "Use 'arcosim <scenario.yml> --image [--record <output.png>]' instead."
)


# ---------------------------------------------------------------------------
# Internal helpers (kept for backward compatibility with existing tests)
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
    print(f"arcoex: loading scenario {path!r}...")

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

    print(f"arcoex: scenario declared: {scenario!r}...")

    if scenario not in SUPPORTED_SCENARIOS:
        print(
            f"arcoex: unknown scenario {scenario!r}. "
            f"Supported values: {sorted(SUPPORTED_SCENARIOS)}",
            file=sys.stderr,
        )
        sys.exit(1)
    return scenario, cfg


# ---------------------------------------------------------------------------
# Public entry point — thin wrapper over _dispatch_static from arcosim
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the ``arcoex`` CLI (deprecated).

    Emits a :class:`DeprecationWarning`, then delegates to
    :func:`arco.tools.arcosim.__main__._dispatch_static`, which provides
    identical functionality via ``arcosim --image``.

    Raises:
        SystemExit: On any validation error or missing dependencies.
    """
    warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=1)
    print(
        f"⚠️  arcoex is deprecated — use 'arcosim --image' instead.",
        file=sys.stderr,
    )

    parser = argparse.ArgumentParser(
        prog="arcoex",
        description=(
            "[DEPRECATED — use 'arcosim --image' instead] "
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
        help=(
            "Save the output image to PATH instead of opening a window. "
            "Equivalent to 'arcosim --image --record PATH'."
        ),
    )
    args = parser.parse_args()

    scenario, cfg = _load_scenario(args.scenario_file)

    # Delegate to arcosim's static dispatch — same code path as
    # 'arcosim <file> --image [--record <path>]'.
    from arco.tools.arcosim.__main__ import _dispatch_static

    _dispatch_static(scenario, cfg, args.save)


if __name__ == "__main__":
    main()
