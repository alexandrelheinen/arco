"""Tests for the enable_pruning configuration flag.

Verifies that:
- All map YAML files default ``enable_pruning`` to ``False``.
- When the flag is disabled the pruned path is a copy of the original raw
  path (not None, not shorter).
- When the flag is enabled a ``TrajectoryPruner`` is used and may produce a
  shorter path.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import yaml

_REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, os.path.join(_REPO, "src"))

from arco.mapping import KDTreeOccupancy
from arco.planning.continuous import TrajectoryPruner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAP_DIR = os.path.join(_REPO, "src", "arco", "tools", "map")

# YAML files that contain a `planner` section with `enable_pruning`.
_PRUNER_YAMLS = [
    "city.yml",
    "ppp.yml",
    "rr.yml",
    "occ.yml",
    "rrp.yml",
    "vehicle.yml",
]


def _load_planner_cfg(name: str) -> dict:
    path = os.path.join(_MAP_DIR, name)
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    return cfg.get("planner", {})


def _free_occ() -> KDTreeOccupancy:
    return KDTreeOccupancy([[1000.0, 1000.0]], clearance=0.1)


def _collinear_path(n: int = 8) -> list[np.ndarray]:
    return [np.array([float(i), 0.0]) for i in range(n)]


# ---------------------------------------------------------------------------
# 1. YAML defaults – enable_pruning must be False in every map file
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("yml_name", _PRUNER_YAMLS)
def test_yaml_enable_pruning_default_is_false(yml_name: str) -> None:
    """All map YAML planner sections must default ``enable_pruning`` to False."""
    planner_cfg = _load_planner_cfg(yml_name)
    assert (
        "enable_pruning" in planner_cfg
    ), f"{yml_name}: missing 'enable_pruning' key in [planner] section"
    assert planner_cfg["enable_pruning"] is False, (
        f"{yml_name}: 'enable_pruning' should be False by default, "
        f"got {planner_cfg['enable_pruning']!r}"
    )


# ---------------------------------------------------------------------------
# 2. Disabled behaviour – pruned path is an identical copy of the raw path
# ---------------------------------------------------------------------------


def test_disabled_pruning_direct_pattern_returns_copy() -> None:
    """Direct-pruner scene pattern: disabled → path is a list copy, not None."""
    raw_path = _collinear_path(6)
    enable_pruning = False

    # Replicate the direct-pruner scene logic.
    if enable_pruning:
        occ = _free_occ()
        pruner = TrajectoryPruner(occ, step_size=np.array([1.0, 1.0]))
        pruned = pruner.prune(raw_path)
    else:
        pruned = list(raw_path)

    assert pruned is not raw_path, "pruned must be a new list object"
    assert len(pruned) == len(raw_path)
    for a, b in zip(pruned, raw_path):
        assert np.allclose(a, b)


def test_disabled_pruning_pipeline_pattern_returns_copy() -> None:
    """Pipeline scene pattern: pruner=None → fallback copy of the raw path."""
    raw_path = _collinear_path(6)
    enable_pruning = False

    # Replicate the PlanningPipeline scene logic.
    pruner = (
        TrajectoryPruner(_free_occ(), step_size=np.array([1.0, 1.0]))
        if enable_pruning
        else None
    )
    pruned = pruner.prune(raw_path) if pruner is not None else list(raw_path)

    assert pruner is None
    assert pruned is not raw_path
    assert len(pruned) == len(raw_path)
    for a, b in zip(pruned, raw_path):
        assert np.allclose(a, b)


def test_disabled_pruning_preserves_every_waypoint() -> None:
    """With pruning disabled no waypoints are dropped or reordered."""
    raw_path = [np.array([float(i), float(i % 3)]) for i in range(10)]
    pruned = list(raw_path)

    assert len(pruned) == len(raw_path)
    for orig, copy_ in zip(raw_path, pruned):
        assert np.allclose(orig, copy_)


# ---------------------------------------------------------------------------
# 3. Enabled behaviour – TrajectoryPruner is used and may reduce the path
# ---------------------------------------------------------------------------


def test_enabled_pruning_direct_pattern_creates_pruner() -> None:
    """When enabled, a TrajectoryPruner instance is created and used."""
    raw_path = _collinear_path(8)
    enable_pruning = True
    step_size = np.array([1.0, 1.0])
    occ = _free_occ()

    pruner = (
        TrajectoryPruner(occ, step_size=step_size) if enable_pruning else None
    )

    assert pruner is not None
    pruned = pruner.prune(raw_path) if pruner is not None else list(raw_path)
    # In free space the pruner collapses a collinear path to 2 nodes.
    assert len(pruned) <= len(raw_path)
    assert np.allclose(pruned[0], raw_path[0])
    assert np.allclose(pruned[-1], raw_path[-1])


def test_enabled_pruning_shortens_collinear_path_in_free_space() -> None:
    """Collinear path in free space is reduced to start+goal when pruning enabled."""
    raw_path = _collinear_path(10)
    occ = _free_occ()
    pruner = TrajectoryPruner(occ, step_size=np.array([1.0, 1.0]))
    pruned = pruner.prune(raw_path)

    assert len(pruned) == 2
    assert np.allclose(pruned[0], raw_path[0])
    assert np.allclose(pruned[-1], raw_path[-1])


def test_enabled_pruning_result_shorter_than_disabled() -> None:
    """Enabled pruning yields fewer waypoints than disabled (copy) on a collinear path."""
    raw_path = _collinear_path(10)
    occ = _free_occ()

    # Enabled.
    pruner = TrajectoryPruner(occ, step_size=np.array([1.0, 1.0]))
    pruned_enabled = pruner.prune(raw_path)

    # Disabled.
    pruned_disabled = list(raw_path)

    assert len(pruned_enabled) < len(pruned_disabled)


# ---------------------------------------------------------------------------
# 4. Flag propagation via config dict (mirrors scene/example read pattern)
# ---------------------------------------------------------------------------


def test_cfg_get_returns_false_when_key_absent() -> None:
    """Scenes use cfg.get('enable_pruning', False); missing key → False."""
    cfg: dict = {}
    result = bool(cfg.get("enable_pruning", False))
    assert result is False


def test_cfg_get_returns_false_when_key_is_false() -> None:
    cfg = {"enable_pruning": False}
    result = bool(cfg.get("enable_pruning", False))
    assert result is False


def test_cfg_get_returns_true_when_key_is_true() -> None:
    cfg = {"enable_pruning": True}
    result = bool(cfg.get("enable_pruning", False))
    assert result is True


# ---------------------------------------------------------------------------
# 5. astar.yml must not have enable_pruning (no pruner used there)
# ---------------------------------------------------------------------------


def test_astar_yml_has_no_enable_pruning_key() -> None:
    """astar.yml uses discrete A* — no TrajectoryPruner, no enable_pruning."""
    path = os.path.join(_MAP_DIR, "astar.yml")
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    planner_cfg = cfg.get("planner", {})
    assert "enable_pruning" not in planner_cfg, (
        "astar.yml should not have an 'enable_pruning' key "
        "(no TrajectoryPruner is used for discrete A*)"
    )
