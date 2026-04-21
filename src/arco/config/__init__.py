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

logger = logging.getLogger(__name__)

# Get the directory of this file, and use it as the default
# config directory if ARCO_CONFIG_DIR is not set
_my_dir = os.path.dirname(os.path.abspath(__file__))
_config_dir = os.getenv("ARCO_CONFIG_DIR", _my_dir)

logger.info("ARCO config dir set to %r", _config_dir)


def load_config(name: str) -> dict[str, Any]:
    """Load a YAML configuration file from ``ARCO_CONFIG_DIR``.

    These are source-code configs accessed directly by library modules
    (planners, vehicles, colors, etc.).  They are **not** scenario files
    and are not intended to be passed to the CLI tools.

    The ARCO root configuration directory must be determined by the
    environment variable ``ARCO_CONFIG_DIR``.  This is expected to be set
    by the user or by the CLI tools, and should point to the directory
    containing the YAML config files.

    Args:
        name: Base name of the config file (without the ``.yml``
            extension), e.g. ``"astar"``, ``"vehicle"``, or ``"colors"``.

    Returns:
        A dictionary containing the parsed YAML configuration.

    Raises:
        EnvironmentError: If the ARCO_CONFIG_DIR environment variable
            is not set.
        FileNotFoundError: If ``ARCO_CONFIG_DIR/<name>.yml``
            does not exist.
    """
    # Look for the specific config file in the directory
    config_path = os.path.join(_config_dir, f"{name}.yml")
    logger.debug("Loading config %r...", config_path)

    with open(config_path) as fh:
        return yaml.safe_load(fh) or {}


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
