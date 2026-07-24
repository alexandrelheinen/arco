"""Tests for the shared arcosim presentation chrome."""

from __future__ import annotations

from arco.simulator.sim.layout import (
    ScreenLayout,
    build_compact_planner_sections,
    scenario_phase_title,
)


def test_screen_layout_content_geometry() -> None:
    layout = ScreenLayout(1280, 720, sidebar_w=280, header_h=44, footer_h=32)
    assert layout.content_x == 280
    assert layout.content_w == 1000
    assert layout.content_h == 720 - 44 - 32
    assert layout.content_y == 32


def test_scenario_phase_title_city_and_arms() -> None:
    assert scenario_phase_title("city", "background") == (
        "City  ·  planning reveal"
    )
    assert scenario_phase_title("city", "racing") == "City  ·  race"
    assert scenario_phase_title("ppp", "show") == "PPP  ·  path reveal"
    assert scenario_phase_title("ppp", "race") == "PPP  ·  race"
    assert scenario_phase_title("rrp", "done") == "RRP  ·  finished"
    assert scenario_phase_title("occ", "running") == (
        "OCC  ·  piano movers  ·  RRT* ‖ SST"
    )


def test_build_compact_planner_sections_keeps_method_colors() -> None:
    sections = build_compact_planner_sections(
        [
            (
                "RRT*",
                (70, 120, 200),
                {
                    "steps": 100,
                    "nodes": 80,
                    "planner_time": 1.2,
                    "planned_path_length": 42.0,
                    "path_status": "ok",
                },
            ),
            (
                "SST",
                (70, 180, 100),
                {
                    "steps": 200,
                    "nodes": 90,
                    "planner_time": 2.0,
                    "planned_path_length": 50.0,
                    "path_status": "ok",
                },
            ),
        ]
    )
    assert len(sections) == 2
    assert sections[0][0][0] == "RRT*"
    assert sections[0][1] == (70, 120, 200)
    assert any("80 nodes" in line for line in sections[0][0])
    assert sections[1][1] == (70, 180, 100)
