"""Utility for loading tool configuration files from ${ARCO_ROOT_DIR}/config/.

Usage::

    from arco.tools.config import load_config

    cfg = load_config("grid")   # loads ${ARCO_ROOT_DIR}/config/grid.yml
    cfg = load_config("map")    # loads ${ARCO_ROOT_DIR}/config/map.yml
    cfg = load_config("random") # loads ${ARCO_ROOT_DIR}/config/random.yml

The configuration directory is determined by the ``ARCO_ROOT_DIR`` environment
variable, which defaults to ``<repo_root>/src/arco/tools``.
All config files are expected to be in ``${ARCO_ROOT_DIR}/config/``.

For custom file systems, set ``ARCO_ROOT_DIR`` to point to your custom root::

    export ARCO_ROOT_DIR=/path/to/custom/root
    # Configs will be loaded from /path/to/custom/root/config/
"""

from __future__ import annotations

import os
from typing import Any

import yaml


def _get_config_dir() -> str:
    """Get the configuration directory path from ARCO_ROOT_DIR environment variable.

    Returns:
        Absolute path to the config directory (${ARCO_ROOT_DIR}/config).
        Defaults to <repo_root>/src/arco/tools if ARCO_ROOT_DIR is not set.
    """
    # Allow override via environment variable
    root_dir = os.environ.get("ARCO_ROOT_DIR")

    if root_dir is None:
        # Default: assume we're in <repo_root>/src/arco/tools/config/__init__.py
        # Go up one level to reach <repo_root>/src/arco/tools (the default ARCO_ROOT_DIR)
        this_file = os.path.abspath(__file__)
        config_dir = os.path.dirname(
            this_file
        )  # <repo_root>/src/arco/tools/config
        root_dir = os.path.dirname(config_dir)  # <repo_root>/src/arco/tools

    return os.path.join(root_dir, "config")


_CONFIG_DIR = _get_config_dir()


def load_config(name: str) -> dict[str, Any]:
    """Load a YAML configuration file from the tools/config directory.

    Args:
        name: Base name of the config file (without the ``.yml``
            extension), e.g. ``"grid"``, ``"map"``, or ``"random"``.

    Returns:
        A dictionary containing the parsed YAML configuration.

    Raises:
        FileNotFoundError: If no matching ``.yml`` file is found.
    """
    path = os.path.join(_CONFIG_DIR, f"{name}.yml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path!r}")
    with open(path) as fh:
        return yaml.safe_load(fh) or {}
