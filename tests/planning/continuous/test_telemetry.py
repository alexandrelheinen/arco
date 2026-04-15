"""Tests for the telemetry module (StopCriterion, PlannerTelemetry, I/O)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from arco.planning.continuous.telemetry import (
    DEFAULT_TELEMETRY_PATH,
    PlannerTelemetry,
    StopCriterion,
    read_telemetry,
    write_telemetry,
)

# ---------------------------------------------------------------------------
# StopCriterion.satisfied()
# ---------------------------------------------------------------------------


def test_stop_criterion_satisfied_less_than():
    c = StopCriterion("iterations", 50.0, 100.0, "<")
    assert c.satisfied() is True

    c_false = StopCriterion("iterations", 100.0, 100.0, "<")
    assert c_false.satisfied() is False


def test_stop_criterion_satisfied_leq():
    c_eq = StopCriterion("dist", 1.0, 1.0, "≤")
    assert c_eq.satisfied() is True

    c_lt = StopCriterion("dist", 0.5, 1.0, "≤")
    assert c_lt.satisfied() is True

    c_gt = StopCriterion("dist", 2.0, 1.0, "≤")
    assert c_gt.satisfied() is False


def test_stop_criterion_satisfied_geq():
    c_eq = StopCriterion("witnesses", 10.0, 10.0, "≥")
    assert c_eq.satisfied() is True

    c_gt = StopCriterion("witnesses", 15.0, 10.0, "≥")
    assert c_gt.satisfied() is True

    c_lt = StopCriterion("witnesses", 5.0, 10.0, "≥")
    assert c_lt.satisfied() is False


def test_stop_criterion_unknown_condition_returns_false():
    c = StopCriterion("x", 1.0, 2.0, "??")
    assert c.satisfied() is False


# ---------------------------------------------------------------------------
# write_telemetry / read_telemetry round-trip
# ---------------------------------------------------------------------------


def test_write_and_read_roundtrip(tmp_path):
    path = tmp_path / "telemetry.json"
    telemetry = PlannerTelemetry(
        algorithm="RRT*",
        step_name="exploring",
        iteration=200,
        max_iterations=2000,
        best_dist_to_goal=3.14,
        criteria=[
            StopCriterion("iterations", 200.0, 2000.0, "<"),
            StopCriterion("dist_to_goal", 3.14, 1.0, "≤"),
        ],
    )
    write_telemetry(telemetry, path)
    result = read_telemetry(path)

    assert result is not None
    assert result.algorithm == "RRT*"
    assert result.step_name == "exploring"
    assert result.iteration == 200
    assert result.max_iterations == 2000
    assert math.isclose(result.best_dist_to_goal, 3.14)
    assert len(result.criteria) == 2
    assert result.criteria[0].name == "iterations"
    assert result.criteria[1].condition == "≤"


# ---------------------------------------------------------------------------
# read_telemetry returns None on missing file
# ---------------------------------------------------------------------------


def test_read_missing_file_returns_none():
    result = read_telemetry(Path("/nonexistent/path.json"))
    assert result is None


# ---------------------------------------------------------------------------
# write_telemetry never raises on bad path
# ---------------------------------------------------------------------------


def test_write_never_raises_on_bad_path():
    telemetry = PlannerTelemetry(
        algorithm="SST",
        step_name="exploring",
        iteration=0,
        max_iterations=1000,
        best_dist_to_goal=math.inf,
        criteria=[],
    )
    # Must not raise even though the directory does not exist.
    write_telemetry(telemetry, Path("/nonexistent/dir/x.json"))
