"""Utility for loading tool configuration files from tools/config/.

Usage::

    from config import load_config

    cfg = load_config("grid")   # loads tools/config/grid.yml
    cfg = load_config("map")    # loads tools/config/map.yml
    cfg = load_config("random") # loads tools/config/random.yml
"""

from __future__ import annotations

import os
from typing import Any

import yaml

_CONFIG_DIR = os.path.dirname(__file__)


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
        raise FileNotFoundError(
            f"Config file not found: {path!r}"
        )
    with open(path) as fh:
        return yaml.safe_load(fh) or {}
