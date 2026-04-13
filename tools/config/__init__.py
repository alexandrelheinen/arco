"""Utility for loading tool configuration files from tools/config/.

Config files are organized in two sub-folders:

- ``map/``    — scenario configs (environment, obstacles, world params).
- ``system/`` — system-level configs (colors, shared parameters).

Usage::

    from config import load_config

    cfg = load_config("astar")   # loads tools/config/map/astar.yml
    cfg = load_config("city")    # loads tools/config/map/city.yml
    cfg = load_config("colors")  # loads tools/config/system/colors.yml
"""

from __future__ import annotations

import os
from typing import Any

import yaml

_CONFIG_DIR = os.path.dirname(__file__)

# Search order: map/ first, then system/, then the legacy root directory.
_SEARCH_DIRS = (
    os.path.join(_CONFIG_DIR, "map"),
    os.path.join(_CONFIG_DIR, "system"),
    _CONFIG_DIR,
)


def load_config(name: str) -> dict[str, Any]:
    """Load a YAML configuration file from the tools/config directory.

    Searches in ``map/``, ``system/``, and the root ``config/`` directory
    (in that order) for a file named ``<name>.yml``.

    Args:
        name: Base name of the config file (without the ``.yml``
            extension), e.g. ``"astar"``, ``"city"``, or ``"colors"``.

    Returns:
        A dictionary containing the parsed YAML configuration.

    Raises:
        FileNotFoundError: If no matching ``.yml`` file is found in any
            of the search directories.
    """
    for directory in _SEARCH_DIRS:
        path = os.path.join(directory, f"{name}.yml")
        if os.path.isfile(path):
            with open(path) as fh:
                return yaml.safe_load(fh) or {}
    searched = ", ".join(_SEARCH_DIRS)
    raise FileNotFoundError(
        f"Config file {name!r} not found. Searched: {searched}"
    )
