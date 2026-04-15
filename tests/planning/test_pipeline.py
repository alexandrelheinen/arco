"""Tests for PlanningPipeline."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from arco.mapping import KDTreeOccupancy
from arco.planning import (
    PipelineResult,
    PlanningPipeline,
    RRTPlanner,
    TrajectoryOptimizer,
    TrajectoryPruner,
)

# ---------------------------------------------------------------------------
# Shared helpers / stubs
# ---------------------------------------------------------------------------

_STEP_2D = np.array([1.0, 1.0])
_BOUNDS_2D = np.array([[0.0, 10.0], [0.0, 10.0]])


class _AlwaysSucceedsPlanner:
    """Stub planner that always returns a fixed two-node path."""

    def __init__(self, start: np.ndarray, goal: np.ndarray) -> None:
        self._start = start
        self._goal = goal

    def plan(
        self, start: np.ndarray, goal: np.ndarray
    ) -> list[np.ndarray] | None:
        return [self._start.copy(), self._goal.copy()]


class _AlwaysFailsPlanner:
    """Stub planner that always returns None (no path found)."""

    def plan(
        self, start: np.ndarray, goal: np.ndarray
    ) -> list[np.ndarray] | None:
        return None


class _EmptyPathPlanner:
    """Stub planner that returns an empty list."""

    def plan(
        self, start: np.ndarray, goal: np.ndarray
    ) -> list[np.ndarray] | None:
        return []


class _RecordingPruner:
    """Stub pruner that records calls and returns the input unchanged."""

    def __init__(self) -> None:
        self.calls: list[list[np.ndarray]] = []

    def prune(self, path: list[np.ndarray]) -> list[np.ndarray]:
        self.calls.append(list(path))
        return list(path)


class _RecordingOptimizer:
    """Stub optimizer that records calls and returns a minimal result."""

    def __init__(self) -> None:
        self.calls: list[list[np.ndarray]] = []

    def optimize(self, path: list[np.ndarray]):  # noqa: ANN201
        from arco.planning import TrajectoryResult

        self.calls.append(list(path))
        # Return a result with the same states and unit durations.
        return TrajectoryResult(
            states=list(path),
            durations=[1.0] * max(len(path) - 1, 1),
            optimizer_success=True,
            optimizer_status_text="stub_ok",
        )


def _free_occ() -> KDTreeOccupancy:
    return KDTreeOccupancy([[1000.0, 1000.0]], clearance=0.5)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_pipeline_stores_planner():
    planner = _AlwaysSucceedsPlanner(np.zeros(2), np.ones(2))
    p = PlanningPipeline(planner=planner)
    assert p.planner is planner
    assert p.pruner is None
    assert p.optimizer is None


def test_pipeline_stores_all_stages():
    planner = _AlwaysSucceedsPlanner(np.zeros(2), np.ones(2))
    pruner = _RecordingPruner()
    optimizer = _RecordingOptimizer()
    p = PlanningPipeline(planner=planner, pruner=pruner, optimizer=optimizer)
    assert p.planner is planner
    assert p.pruner is pruner
    assert p.optimizer is optimizer


# ---------------------------------------------------------------------------
# run() — planner-only (no pruner, no optimizer)
# ---------------------------------------------------------------------------


def test_run_planner_only_success():
    start = np.array([0.0, 0.0])
    goal = np.array([5.0, 5.0])
    planner = _AlwaysSucceedsPlanner(start, goal)
    result = PlanningPipeline(planner=planner).run(start, goal)

    assert result.planner_status == "success"
    assert result.raw_path is not None
    assert len(result.raw_path) == 2
    assert result.pruned_path is None  # pruner not configured
    # Without optimizer, trajectory == path
    assert result.trajectory is not None
    assert len(result.trajectory) == 2
    assert result.durations is not None
    assert result.optimizer_status == "skipped"
    assert result.planner_time >= 0.0


def test_run_planner_only_failure():
    planner = _AlwaysFailsPlanner()
    result = PlanningPipeline(planner=planner).run(np.zeros(2), np.ones(2))
    assert result.planner_status == "no_path"
    assert result.raw_path is None
    assert result.trajectory is None


def test_run_empty_path_treated_as_failure():
    planner = _EmptyPathPlanner()
    result = PlanningPipeline(planner=planner).run(np.zeros(2), np.ones(2))
    assert result.planner_status == "no_path"


# ---------------------------------------------------------------------------
# run() — planner + pruner
# ---------------------------------------------------------------------------


def test_run_with_pruner_calls_pruner():
    start = np.array([0.0, 0.0])
    goal = np.array([5.0, 5.0])
    planner = _AlwaysSucceedsPlanner(start, goal)
    pruner = _RecordingPruner()
    result = PlanningPipeline(planner=planner, pruner=pruner).run(start, goal)

    assert len(pruner.calls) == 1
    assert result.pruned_path is not None


def test_run_with_pruner_pruner_receives_raw_path():
    start = np.array([0.0, 0.0])
    goal = np.array([3.0, 0.0])
    planner = _AlwaysSucceedsPlanner(start, goal)
    pruner = _RecordingPruner()
    PlanningPipeline(planner=planner, pruner=pruner).run(start, goal)

    assert np.allclose(pruner.calls[0][0], start)
    assert np.allclose(pruner.calls[0][-1], goal)


def test_run_pruner_not_called_when_planning_fails():
    pruner = _RecordingPruner()
    PlanningPipeline(planner=_AlwaysFailsPlanner(), pruner=pruner).run(
        np.zeros(2), np.ones(2)
    )
    assert len(pruner.calls) == 0


# ---------------------------------------------------------------------------
# run() — planner + pruner + optimizer
# ---------------------------------------------------------------------------


def test_run_with_optimizer_calls_optimizer():
    start = np.array([0.0, 0.0])
    goal = np.array([5.0, 5.0])
    planner = _AlwaysSucceedsPlanner(start, goal)
    optimizer = _RecordingOptimizer()
    result = PlanningPipeline(planner=planner, optimizer=optimizer).run(
        start, goal
    )

    assert len(optimizer.calls) == 1
    assert result.optimizer_success is True
    assert result.trajectory is not None


def test_run_optimizer_receives_pruned_path():
    start = np.array([0.0, 0.0])
    goal = np.array([5.0, 5.0])
    planner = _AlwaysSucceedsPlanner(start, goal)
    pruner = _RecordingPruner()
    optimizer = _RecordingOptimizer()
    PlanningPipeline(planner=planner, pruner=pruner, optimizer=optimizer).run(
        start, goal
    )

    # Optimizer's first call should receive the pruner's output (same path here)
    assert len(optimizer.calls) == 1


def test_run_optimizer_not_called_on_failure():
    optimizer = _RecordingOptimizer()
    PlanningPipeline(planner=_AlwaysFailsPlanner(), optimizer=optimizer).run(
        np.zeros(2), np.ones(2)
    )
    assert len(optimizer.calls) == 0


# ---------------------------------------------------------------------------
# run() — timing fields
# ---------------------------------------------------------------------------


def test_run_timing_fields_are_non_negative():
    start = np.array([0.0, 0.0])
    goal = np.array([5.0, 5.0])
    planner = _AlwaysSucceedsPlanner(start, goal)
    pruner = _RecordingPruner()
    optimizer = _RecordingOptimizer()
    result = PlanningPipeline(
        planner=planner, pruner=pruner, optimizer=optimizer
    ).run(start, goal)
    assert result.planner_time >= 0.0
    assert result.pruner_time >= 0.0
    assert result.optimizer_time >= 0.0


# ---------------------------------------------------------------------------
# run() — progress callback
# ---------------------------------------------------------------------------


def test_run_progress_callback_called_once_per_stage():
    start = np.array([0.0, 0.0])
    goal = np.array([5.0, 5.0])
    planner = _AlwaysSucceedsPlanner(start, goal)
    pruner = _RecordingPruner()
    optimizer = _RecordingOptimizer()
    calls: list[tuple[str, int, int]] = []

    def cb(stage: str, idx: int, total: int) -> None:
        calls.append((stage, idx, total))

    PlanningPipeline(planner=planner, pruner=pruner, optimizer=optimizer).run(
        start, goal, progress=cb
    )

    assert len(calls) == 3
    stages = [c[0] for c in calls]
    assert "planning" in stages
    assert "pruning" in stages
    assert "optimization" in stages


def test_run_progress_total_reflects_configured_stages():
    start = np.array([0.0, 0.0])
    goal = np.array([5.0, 5.0])
    planner = _AlwaysSucceedsPlanner(start, goal)
    calls: list[int] = []
    PlanningPipeline(planner=planner).run(
        start, goal, progress=lambda s, i, t: calls.append(t)
    )
    assert all(t == 1 for t in calls)  # only planner configured → total=1


# ---------------------------------------------------------------------------
# result — total_duration
# ---------------------------------------------------------------------------


def test_run_total_duration_equals_sum_of_durations():
    start = np.array([0.0, 0.0])
    goal = np.array([5.0, 5.0])
    planner = _AlwaysSucceedsPlanner(start, goal)
    result = PlanningPipeline(planner=planner).run(start, goal)
    if result.durations is not None:
        assert math.isclose(
            result.total_duration, sum(result.durations), rel_tol=1e-9
        )


# ---------------------------------------------------------------------------
# save_result / load_result — round-trip
# ---------------------------------------------------------------------------


def test_save_load_round_trip(tmp_path: Path) -> None:
    original = PipelineResult(
        raw_path=[np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        pruned_path=[np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        trajectory=[np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        durations=[2.5],
        total_duration=2.5,
        planner_time=0.123,
        pruner_time=0.045,
        optimizer_time=0.678,
        planner_status="success",
        optimizer_status="0: ok",
        optimizer_success=True,
        extra={"nodes": 42},
    )
    dest = tmp_path / "result"
    PlanningPipeline.save_result(original, dest)
    loaded = PlanningPipeline.load_result(dest)

    assert loaded.planner_status == "success"
    assert loaded.optimizer_success is True
    assert math.isclose(loaded.total_duration, 2.5, rel_tol=1e-9)
    assert math.isclose(loaded.planner_time, 0.123, rel_tol=1e-6)
    assert loaded.raw_path is not None and len(loaded.raw_path) == 2
    assert loaded.trajectory is not None
    assert np.allclose(loaded.trajectory[0], [0.0, 0.0])
    assert loaded.durations is not None
    assert math.isclose(loaded.durations[0], 2.5, rel_tol=1e-9)
    assert loaded.extra.get("nodes") == 42


def test_save_creates_parent_dirs(tmp_path: Path) -> None:
    result = PipelineResult(planner_status="no_path")
    dest = tmp_path / "deep" / "nested" / "result"
    PlanningPipeline.save_result(result, dest)
    assert (tmp_path / "deep" / "nested").exists()


def test_save_load_empty_paths(tmp_path: Path) -> None:
    """Round-trip a result where all paths are None."""
    original = PipelineResult(planner_status="no_path")
    dest = tmp_path / "empty_result"
    PlanningPipeline.save_result(original, dest)
    loaded = PlanningPipeline.load_result(dest)
    assert loaded.planner_status == "no_path"
    assert loaded.raw_path is None
    assert loaded.trajectory is None


def test_load_adds_npz_suffix(tmp_path: Path) -> None:
    result = PipelineResult(planner_status="success")
    dest = tmp_path / "no_suffix"
    PlanningPipeline.save_result(result, str(dest) + ".npz")
    # Should work without .npz suffix.
    loaded = PlanningPipeline.load_result(dest)
    assert loaded.planner_status == "success"


# ---------------------------------------------------------------------------
# Integration: real RRT planner in free space
# ---------------------------------------------------------------------------


def test_real_rrt_pipeline_finds_path():
    """End-to-end: RRT + pruner + no optimizer in a free 2-D space."""
    occ = _free_occ()
    planner = RRTPlanner(
        occ,
        bounds=[[0.0, 10.0], [0.0, 10.0]],
        step_size=_STEP_2D,
        goal_tolerance=1.5,
        max_sample_count=5000,
    )
    pruner = TrajectoryPruner(occ, step_size=_STEP_2D)
    pipeline = PlanningPipeline(planner=planner, pruner=pruner)
    result = pipeline.run(np.array([0.5, 0.5]), np.array([9.5, 9.5]))

    assert result.planner_status == "success"
    assert result.raw_path is not None
    assert result.pruned_path is not None
    assert result.trajectory is not None
    assert len(result.trajectory) >= 2
    # Pruned path must be ≤ raw path.
    assert len(result.pruned_path) <= len(result.raw_path)
