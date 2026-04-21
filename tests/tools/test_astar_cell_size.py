"""Tests for the A* grid cell size configuration.

Verifies that:
- ``_coerce_astar_cell_size`` defaults to ``step_size / 5`` when no explicit
  cell size is given.
- An explicit ``cell_size`` override is honored (e.g. from ``astar_cell_size``
  in ``city.yml``).
- Both scalar and list/tuple/ndarray ``step_size`` inputs are handled.
- The ``city.yml`` planner section has an ``astar_cell_size`` key equal to
  ``step_size[0] / 5``.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import yaml

pygame = pytest.importorskip("pygame")

_REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, os.path.join(_REPO, "src"))

from arco.simulator.scenes.sparse import _coerce_astar_cell_size

_MAP_DIR = os.path.join(_REPO, "src", "arco", "tools", "map")


# ---------------------------------------------------------------------------
# 1. Default: step_size / 5
# ---------------------------------------------------------------------------


def test_scalar_step_size_defaults_to_fifth() -> None:
    assert _coerce_astar_cell_size(15.0) == pytest.approx(3.0)


def test_list_step_size_defaults_to_fifth() -> None:
    assert _coerce_astar_cell_size([15.0, 15.0]) == pytest.approx(3.0)


def test_tuple_step_size_defaults_to_fifth() -> None:
    assert _coerce_astar_cell_size((10.0, 10.0)) == pytest.approx(2.0)


def test_ndarray_step_size_defaults_to_fifth() -> None:
    assert _coerce_astar_cell_size(np.array([20.0, 20.0])) == pytest.approx(
        4.0
    )


def test_empty_sequence_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        _coerce_astar_cell_size([])


# ---------------------------------------------------------------------------
# 2. Explicit cell_size override
# ---------------------------------------------------------------------------


def test_explicit_cell_size_overrides_default() -> None:
    assert _coerce_astar_cell_size(15.0, cell_size=5.0) == pytest.approx(5.0)


def test_explicit_cell_size_with_list_step_size() -> None:
    assert _coerce_astar_cell_size(
        [15.0, 15.0], cell_size=2.5
    ) == pytest.approx(2.5)


def test_explicit_cell_size_none_falls_back_to_default() -> None:
    assert _coerce_astar_cell_size(15.0, cell_size=None) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 3. city.yml has astar_cell_size == step_size[0] / 5
# ---------------------------------------------------------------------------


def test_city_yml_has_astar_cell_size() -> None:
    """city.yml must declare astar_cell_size in the planner section."""
    path = os.path.join(_MAP_DIR, "city.yml")
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    planner = cfg.get("planner", {})
    assert (
        "astar_cell_size" in planner
    ), "city.yml planner section must have an 'astar_cell_size' key"


def test_city_yml_astar_cell_size_equals_step_size_over_five() -> None:
    """city.yml astar_cell_size must equal step_size[0] / 5."""
    path = os.path.join(_MAP_DIR, "city.yml")
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    planner = cfg["planner"]
    step_size = planner["step_size"]
    if isinstance(step_size, (list, tuple)):
        base = float(step_size[0])
    else:
        base = float(step_size)
    expected = base / 5.0
    assert float(planner["astar_cell_size"]) == pytest.approx(expected), (
        f"astar_cell_size should be {expected} (step_size[0]/5) but got "
        f"{planner['astar_cell_size']}"
    )
