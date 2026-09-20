"""Tests for the arcosim CLI fast-record / dispatch wiring.

These tests must not import pygame: CI unit jobs install ``arco`` without the
display extras.  ``arco.simulator.__main__`` lazy-imports scenario mains.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from arco.simulator import __main__ as arcosim_main


def test_parse_args_fast_record_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "arcosim",
            "map/city.yml",
            "-o",
            "/tmp/x.mp4",
            "-d",
            "30",
            "--fast-record",
        ],
    )
    args = arcosim_main.parse_args()
    assert args.scenario_file == "map/city.yml"
    assert args.output == "/tmp/x.mp4"
    assert abs(args.record_duration - 30.0) < 1e-12
    assert args.fast_record is True


def test_apply_fast_record_sets_simulator_flag() -> None:
    cfg: dict[str, Any] = {"scenario": "city", "planner": {}}
    arcosim_main._apply_fast_record(cfg)
    assert cfg["simulator"]["fast_record"] is True


def test_dispatch_sets_simulator_fast_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class _FakeCity:
        @staticmethod
        def main(
            cfg: dict[str, Any], save_path: str | None, duration: float
        ) -> None:
            captured["cfg"] = cfg
            captured["save_path"] = save_path
            captured["duration"] = duration

    fake_main = types.ModuleType("arco.simulator.main")
    fake_main.city = _FakeCity()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "arco.simulator.main", fake_main)

    cfg: dict[str, Any] = {"scenario": "city", "planner": {}}
    arcosim_main._dispatch(cfg, "/tmp/out.mp4", 45.0, fast_record=True)
    assert cfg["simulator"]["fast_record"] is True
    assert captured["save_path"] == "/tmp/out.mp4"
    assert abs(captured["duration"] - 45.0) < 1e-12


# ---------------------------------------------------------------------------
# Still-frame flags
# ---------------------------------------------------------------------------


# AC-STILL-01
def test_parse_args_still_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "arcosim",
            "map/city.yml",
            "-o",
            "/tmp/city.png",
            "--still",
            "900",
            "--width",
            "1920",
            "--height",
            "1080",
            "--seed",
            "7",
        ],
    )
    args = arcosim_main.parse_args()
    assert args.still == "900"
    assert args.width == 1920
    assert args.height == 1080
    assert args.seed == 7


# AC-STILL-04
def test_parse_args_without_still_leaves_the_flags_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys, "argv", ["arcosim", "map/city.yml", "-o", "/tmp/x.mp4"]
    )
    args = arcosim_main.parse_args()
    assert args.still is None
    assert args.width is None
    assert args.height is None
    assert args.seed is None


# AC-STILL-02
def test_build_still_request_parses_several_frames(tmp_path: Any) -> None:
    request = arcosim_main._build_still_request(
        still_spec="300,900,300",
        output=str(tmp_path / "city.png"),
        width=1920,
        height=1080,
    )
    assert request.frames == (300, 900)
    assert request.width == 1920
    assert request.height == 1080
    assert request.last_frame == 900


# AC-STILL-01
def test_build_still_request_rejects_a_missing_output() -> None:
    with pytest.raises(SystemExit):
        arcosim_main._build_still_request(
            still_spec="900", output=None, width=None, height=None
        )


# AC-STILL-05
def test_still_duration_covers_the_last_requested_frame() -> None:
    # 900 frames at 30 fps is 30 s of recording; the default 360 s budget
    # already covers it, while frame 20000 needs a longer run.
    assert arcosim_main._still_duration(900, 360.0, fps=30) == 360.0
    assert arcosim_main._still_duration(20000, 360.0, fps=30) > 666.0
