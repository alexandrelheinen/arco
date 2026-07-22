"""Utility for loading package configuration files.

:func:`load_config` loads YAML files from ``arco/config/`` (palette,
optimizer defaults, MPC defaults, etc.). Scenario files for ``arcosim`` live
in the repository ``map/`` directory and are loaded by the CLI, not by this
module.

Usage::

    from arco.config import load_config

    cfg = load_config("colors")      # loads arco/config/colors.yml
    cfg = load_config("optimizer")   # loads arco/config/optimizer.yml
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
            extension), e.g. ``"optimizer"``, ``"mpc"``, or ``"colors"``.

    Returns:
        A dictionary containing the parsed YAML configuration.

    Raises:
        FileNotFoundError: If ``ARCO_CONFIG_DIR/<name>.yml`` does not exist.
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
