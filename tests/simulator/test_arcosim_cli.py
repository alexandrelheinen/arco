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
