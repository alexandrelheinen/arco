"""Tests for SceneSnapshot JSON parser and data model."""

from __future__ import annotations

import json

import pytest

from arco.simulator.viewer.scene_snapshot import SceneSnapshot

# ---------------------------------------------------------------------------
# Construction and defaults
# ---------------------------------------------------------------------------


def test_default_snapshot_has_empty_collections() -> None:
    snap = SceneSnapshot()
    assert snap.scenario == ""
    assert snap.planner == ""
    assert snap.start == []
    assert snap.goal == []
    assert snap.obstacles == []
    assert snap.tree_nodes == []
    assert snap.tree_parent == []
    assert snap.found_path is None
    assert snap.pruned_path is None
    assert snap.adjusted_trajectory is None
    assert snap.executed_trajectory is None
    assert snap.metrics == {}


def test_from_planning_result_sets_all_fields() -> None:
    snap = SceneSnapshot.from_planning_result(
        scenario="rr",
        planner="rrt",
        start=[0.0, 0.0],
        goal=[1.57, -0.5],
        obstacles=[[-1.0, -1.0, 1.0, 1.0]],
        tree_nodes=[[0.0, 0.0], [0.5, 0.2]],
        tree_parent=[-1, 0],
        found_path=[[0.0, 0.0], [1.57, -0.5]],
        pruned_path=[[0.0, 0.0], [1.57, -0.5]],
        adjusted_trajectory=[[0.0, 0.0], [1.57, -0.5]],
        executed_trajectory=[[0.0, 0.0], [0.8, -0.25], [1.57, -0.5]],
        metrics={"plan_time": 1.2, "path_length": 1.7},
    )
    assert snap.scenario == "rr"
    assert snap.planner == "rrt"
    assert snap.start == [0.0, 0.0]
    assert snap.obstacles == [[-1.0, -1.0, 1.0, 1.0]]
    assert snap.tree_parent == [-1, 0]
    assert snap.found_path is not None
    assert len(snap.executed_trajectory) == 3  # type: ignore[arg-type]
    assert snap.metrics["plan_time"] == pytest.approx(1.2)


def test_from_planning_result_defaults_none_to_empty_list() -> None:
    snap = SceneSnapshot.from_planning_result(
        scenario="test",
        planner="sst",
        start=[0.0],
        goal=[1.0],
    )
    assert snap.obstacles == []
    assert snap.tree_nodes == []
    assert snap.tree_parent == []
    assert snap.metrics == {}


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------


def test_to_dict_returns_plain_dict() -> None:
    snap = SceneSnapshot(scenario="ppp", planner="rrt", start=[1.0, 2.0])
    d = snap.to_dict()
    assert isinstance(d, dict)
    assert d["scenario"] == "ppp"
    assert d["start"] == [1.0, 2.0]


def test_to_dict_none_values_preserved() -> None:
    snap = SceneSnapshot()
    d = snap.to_dict()
    assert d["found_path"] is None
    assert d["executed_trajectory"] is None


# ---------------------------------------------------------------------------
# to_json / from_json round-trip
# ---------------------------------------------------------------------------


def test_json_round_trip_preserves_all_fields() -> None:
    original = SceneSnapshot.from_planning_result(
        scenario="rrp",
        planner="sst",
        start=[0.1, -0.2, 0.5],
        goal=[1.0, 0.5, 1.5],
        found_path=[[0.1, -0.2, 0.5], [1.0, 0.5, 1.5]],
        metrics={"elapsed": 3.14, "nodes": 200},
    )
    restored = SceneSnapshot.from_json(original.to_json())
    assert restored.scenario == original.scenario
    assert restored.planner == original.planner
    assert restored.start == original.start
    assert restored.goal == original.goal
    assert restored.found_path == original.found_path
    assert restored.metrics["elapsed"] == pytest.approx(3.14)
    assert restored.metrics["nodes"] == 200


def test_to_json_produces_valid_json() -> None:
    snap = SceneSnapshot(scenario="city", planner="rrt")
    text = snap.to_json()
    parsed = json.loads(text)
    assert parsed["scenario"] == "city"


def test_to_json_compact_mode() -> None:
    snap = SceneSnapshot(scenario="x")
    compact = snap.to_json(indent=None)
    assert "\n" not in compact


def test_from_json_ignores_unknown_keys() -> None:
    data = {
        "scenario": "astar",
        "planner": "rrt",
        "start": [],
        "goal": [],
        "UNKNOWN_FUTURE_KEY": "value",
    }
    snap = SceneSnapshot.from_json(json.dumps(data))
    assert snap.scenario == "astar"


def test_from_json_raises_on_non_object() -> None:
    with pytest.raises(TypeError):
        SceneSnapshot.from_json(json.dumps([1, 2, 3]))


def test_from_json_raises_on_invalid_json() -> None:
    with pytest.raises(json.JSONDecodeError):
        SceneSnapshot.from_json("not valid json {{{")


# ---------------------------------------------------------------------------
# from_dict
# ---------------------------------------------------------------------------


def test_from_dict_partial_populates_defaults() -> None:
    snap = SceneSnapshot.from_dict({"scenario": "occ", "planner": "sst"})
    assert snap.scenario == "occ"
    assert snap.start == []


# ---------------------------------------------------------------------------
# Importable from viewer package
# ---------------------------------------------------------------------------


def test_scene_snapshot_importable_from_viewer() -> None:
    from arco.simulator.viewer import SceneSnapshot as SS

    assert SS is SceneSnapshot
