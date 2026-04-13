"""Tests for arcoex and arcosim CLI utility functions."""

from __future__ import annotations

import os
import sys

import pytest
import yaml

# Make arco.tools importable even if not pip-installed in this test run.
_REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, os.path.join(_REPO, "src"))

from arco.tools.arcoex.__main__ import SUPPORTED_SCENARIOS as ARCOEX_SCENARIOS
from arco.tools.arcoex.__main__ import _load_scenario as _arcoex_load
from arco.tools.arcosim.__main__ import (
    SUPPORTED_SCENARIOS as ARCOSIM_SCENARIOS,
)
from arco.tools.arcosim.__main__ import _load_scenario as _arcosim_load

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_CONFIG_MAP = os.path.join(_REPO, "src", "arco", "tools", "config", "map")


def _scenario_yml(name: str) -> str:
    return os.path.join(_CONFIG_MAP, f"{name}.yml")


# ---------------------------------------------------------------------------
# SUPPORTED_SCENARIOS
# ---------------------------------------------------------------------------


def test_arcoex_supported_scenarios_non_empty() -> None:
    assert len(ARCOEX_SCENARIOS) > 0


def test_arcosim_supported_scenarios_non_empty() -> None:
    assert len(ARCOSIM_SCENARIOS) > 0


def test_arcoex_and_arcosim_share_same_scenarios() -> None:
    assert ARCOEX_SCENARIOS == ARCOSIM_SCENARIOS


# ---------------------------------------------------------------------------
# _load_scenario — valid YAML files
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", sorted(ARCOEX_SCENARIOS))
def test_arcoex_load_scenario_shipped_configs(scenario: str) -> None:
    """Every shipped config/map/*.yml is loadable by arcoex."""
    path = _scenario_yml(scenario)
    name, cfg = _arcoex_load(path)
    assert name == scenario
    assert isinstance(cfg, dict)
    assert cfg.get("scenario") == scenario


@pytest.mark.parametrize("scenario", sorted(ARCOSIM_SCENARIOS))
def test_arcosim_load_scenario_shipped_configs(scenario: str) -> None:
    """Every shipped config/map/*.yml is loadable by arcosim."""
    path = _scenario_yml(scenario)
    name, cfg = _arcosim_load(path)
    assert name == scenario
    assert isinstance(cfg, dict)
    assert cfg.get("scenario") == scenario


# ---------------------------------------------------------------------------
# _load_scenario — error cases
# ---------------------------------------------------------------------------


def test_arcoex_load_missing_file_exits(
    tmp_path: pytest.TempPathFactory,
) -> None:
    with pytest.raises(SystemExit) as exc:
        _arcoex_load(str(tmp_path / "does_not_exist.yml"))
    assert exc.value.code != 0


def test_arcosim_load_missing_file_exits(
    tmp_path: pytest.TempPathFactory,
) -> None:
    with pytest.raises(SystemExit) as exc:
        _arcosim_load(str(tmp_path / "does_not_exist.yml"))
    assert exc.value.code != 0


def test_arcoex_load_missing_scenario_key_exits(
    tmp_path: pytest.TempPathFactory,
) -> None:
    yml = tmp_path / "bad.yml"
    yml.write_text("grid:\n  cell_size: 1.0\n")
    with pytest.raises(SystemExit) as exc:
        _arcoex_load(str(yml))
    assert exc.value.code != 0


def test_arcosim_load_missing_scenario_key_exits(
    tmp_path: pytest.TempPathFactory,
) -> None:
    yml = tmp_path / "bad.yml"
    yml.write_text("grid:\n  cell_size: 1.0\n")
    with pytest.raises(SystemExit) as exc:
        _arcosim_load(str(yml))
    assert exc.value.code != 0


def test_arcoex_load_unknown_scenario_exits(
    tmp_path: pytest.TempPathFactory,
) -> None:
    yml = tmp_path / "unknown.yml"
    yml.write_text("scenario: nonexistent\n")
    with pytest.raises(SystemExit) as exc:
        _arcoex_load(str(yml))
    assert exc.value.code != 0


def test_arcosim_load_unknown_scenario_exits(
    tmp_path: pytest.TempPathFactory,
) -> None:
    yml = tmp_path / "unknown.yml"
    yml.write_text("scenario: nonexistent\n")
    with pytest.raises(SystemExit) as exc:
        _arcosim_load(str(yml))
    assert exc.value.code != 0


# ---------------------------------------------------------------------------
# Config map directory — structural contract
# ---------------------------------------------------------------------------


def test_config_map_contains_only_yml_and_json() -> None:
    """config/map/ must contain only .yml and .json files (no sub-folders)."""
    entries = os.listdir(_CONFIG_MAP)
    for entry in entries:
        full = os.path.join(_CONFIG_MAP, entry)
        assert os.path.isfile(
            full
        ), f"Unexpected subdirectory in config/map/: {entry}"
        assert entry.endswith(
            (".yml", ".json")
        ), f"Unexpected file type in config/map/: {entry}"


def test_config_system_contains_colors() -> None:
    system_dir = os.path.join(
        _REPO, "src", "arco", "tools", "config", "system"
    )
    assert os.path.isfile(os.path.join(system_dir, "colors.yml"))


def test_config_dir_contains_only_map_and_system() -> None:
    """config/ must contain only map/, system/, and __init__.py."""
    config_dir = os.path.join(_REPO, "src", "arco", "tools", "config")
    allowed = {"map", "system", "__init__.py", "__pycache__"}
    for entry in os.listdir(config_dir):
        assert entry in allowed, (
            f"Unexpected entry in config/: {entry!r}. "
            "config/ must contain only map/, system/, and __init__.py."
        )


def test_pyproject_includes_tool_config_package_data() -> None:
    """Wheel metadata must include tool config YAML and JSON files."""
    pyproject = os.path.join(_REPO, "pyproject.toml")
    with open(pyproject, encoding="utf-8") as fh:
        content = fh.read()

    assert '[tool.setuptools.package-data]' in content
    assert '"arco.tools.config" = [' in content
    assert '"map/*.yml"' in content
    assert '"map/*.json"' in content
    assert '"system/*.yml"' in content
