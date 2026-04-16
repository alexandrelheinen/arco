"""Utility for loading tool configuration files.

Two loader functions are provided:

- :func:`load_config` — loads source-code configs from ``arco/config/``
  (renderer color palette, etc.).
- :func:`load_map_config` — loads scenario files from ``arco/tools/map/``
  (full scenario YAMLs launched by ``arcosim``).

Usage::

    from arco.config import load_config, load_map_config

    cfg = load_config("colors")      # loads arco/config/colors.yml
    cfg = load_map_config("rr")      # loads arco/tools/map/rr.yml
    cfg = load_map_config("vehicle") # loads arco/tools/map/vehicle.yml
"""

from __future__ import annotations

import logging
import os
from typing import Any

import yaml

_CONFIG_DIR = os.getenv(
    "ARCO_CONFIG_DIR", os.path.join(os.path.dirname(__file__))
)
_MAP_DIR = os.path.join(os.path.dirname(__file__), "..", "tools", "map")

logger = logging.getLogger(__name__)


def _load_yaml(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file does not exist: {path!r}")
    logger.info("Loading config %r...", path)
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def load_config(name: str) -> dict[str, Any]:
    """Load a YAML configuration file from ``tools/config/``.

    These are source-code configs accessed directly by library modules
    (planners, vehicles, colors, etc.).  They are **not** scenario files
    and are not intended to be passed to the CLI tools.

    Args:
        name: Base name of the config file (without the ``.yml``
            extension), e.g. ``"astar"``, ``"vehicle"``, or ``"colors"``.

    Returns:
        A dictionary containing the parsed YAML configuration.

    Raises:
        FileNotFoundError: If ``tools/config/<name>.yml`` does not exist.
    """
    return _load_yaml(os.path.join(_CONFIG_DIR, f"{name}.yml"))


def load_map_config(name: str) -> dict[str, Any]:
    """Load a scenario YAML file from ``tools/map/``.

    These are full scenario configs launched by ``arcosim`` — they carry a
    ``scenario:`` key at the top.  Simulator and example modules use this
    function to read scenario-specific parameters.

    Args:
        name: Base name of the scenario file (without the ``.yml``
            extension), e.g. ``"rr"``, ``"rrp"``, or ``"occ"``.

    Returns:
        A dictionary containing the parsed YAML configuration.

    Raises:
        FileNotFoundError: If ``tools/map/<name>.yml`` does not exist.
    """
    return _load_yaml(os.path.join(_MAP_DIR, f"{name}.yml"))


from arco.config.palette import (  # noqa: E402
    LAYER_ALPHA,
    annotation_hex,
    annotation_rgb,
    hex_to_float,
    hex_to_rgb,
    layer_float,
    layer_hex,
    layer_rgb,
    method_base_float,
    method_base_hex,
    method_base_rgb,
    obstacle_float,
    obstacle_hex,
    obstacle_rgb,
    ui_rgb,
)

__all__ = [
    "load_config",
    "load_map_config",
    "LAYER_ALPHA",
    "annotation_hex",
    "annotation_rgb",
    "hex_to_float",
    "hex_to_rgb",
    "layer_float",
    "layer_hex",
    "layer_rgb",
    "method_base_float",
    "method_base_hex",
    "method_base_rgb",
    "obstacle_float",
    "obstacle_hex",
    "obstacle_rgb",
    "ui_rgb",
]
