"""Config loader for arco.tools.

Config files are loaded from ``${ARCO_ROOT_DIR}/config/`` when the
``ARCO_ROOT_DIR`` environment variable is set, or from the package-bundled
``config/`` directory when the variable is absent.
"""

from __future__ import annotations

import os
from typing import Any

import yaml

_PKG_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config(name: str) -> dict[str, Any]:
    """Load a YAML configuration file.

    Looks in ``${ARCO_ROOT_DIR}/config/`` when the ``ARCO_ROOT_DIR``
    environment variable is set; otherwise falls back to the
    package-bundled config directory.

    Args:
        name: Base name of the config file (without the ``.yml``
            extension), e.g. ``"grid"``, ``"planners"``, or ``"vehicle"``.

    Returns:
        A dictionary containing the parsed YAML configuration.

    Raises:
        FileNotFoundError: If no matching ``.yml`` file is found.
    """
    root = os.environ.get("ARCO_ROOT_DIR")
    config_dir = os.path.join(root, "config") if root else _PKG_CONFIG_DIR
    path = os.path.join(config_dir, f"{name}.yml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path!r}")
    with open(path) as fh:
        return yaml.safe_load(fh) or {}
