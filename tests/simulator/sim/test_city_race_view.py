"""Tests for city race follow-cam framing and standings HUD."""

from __future__ import annotations

from arco.simulator.sim.city_race_view import (
    RACE_VIEW_MIN_HALF_EXTENT,
    build_race_standings,
    pack_centroid,
    race_view_bounds,
)


def test_phase_chrome_title_matches_shared_helper() -> None:
    from arco.simulator.sim.city_race_view import phase_chrome_title
    from arco.simulator.sim.layout import scenario_phase_title

    assert phase_chrome_title("racing") == scenario_phase_title(
        "city", "racing"
    )


def test_pack_centroid_is_mean_of_positions() -> None:
    cx, cy = pack_centroid([(0.0, 0.0), (10.0, 20.0), (20.0, 10.0)])
    assert abs(cx - 10.0) < 1e-12
    assert abs(cy - 10.0) < 1e-12


def test_pack_centroid_empty_falls_back_to_origin() -> None:
    assert pack_centroid([]) == (0.0, 0.0)


def test_race_view_bounds_enforce_min_half_extent() -> None:
    """A tight pack still gets a readable zoomed window, not a pinhead."""
    x0, x1, y0, y1 = race_view_bounds(
        100.0,
        100.0,
        [(98.0, 99.0), (102.0, 101.0)],
        world_x_min=0.0,
        world_x_max=600.0,
        world_y_min=0.0,
        world_y_max=600.0,
    )
    assert abs((x1 - x0) / 2.0 - RACE_VIEW_MIN_HALF_EXTENT) < 1e-9
    assert abs((y1 - y0) / 2.0 - RACE_VIEW_MIN_HALF_EXTENT) < 1e-9
    assert abs((x0 + x1) / 2.0 - 100.0) < 1e-9
    assert abs((y0 + y1) / 2.0 - 100.0) < 1e-9


def test_race_view_bounds_expand_to_fit_spread_pack() -> None:
    x0, x1, y0, y1 = race_view_bounds(
        200.0,
        200.0,
        [(100.0, 200.0), (300.0, 200.0)],
        world_x_min=0.0,
        world_x_max=600.0,
        world_y_min=0.0,
        world_y_max=600.0,
        min_half_extent=80.0,
        margin=40.0,
    )
    # span_x=200 → half = 100 + 40 = 140 > min 80
    assert (x1 - x0) / 2.0 >= 140.0 - 1e-9
    assert x0 >= 0.0 - 1e-9
    assert x1 <= 600.0 + 1e-9


def test_race_view_bounds_clamp_near_world_edge() -> None:
    x0, x1, y0, y1 = race_view_bounds(
        10.0,
        10.0,
        [(5.0, 5.0)],
        world_x_min=0.0,
        world_x_max=600.0,
        world_y_min=0.0,
        world_y_max=600.0,
        min_half_extent=100.0,
        margin=0.0,
    )
    assert x0 >= 0.0 - 1e-9
    assert y0 >= 0.0 - 1e-9
    assert x1 - x0 == 200.0
    assert y1 - y0 == 200.0


def test_build_race_standings_orders_finishers_then_active() -> None:
    sections = build_race_standings(
        [
            ("RRT*", (70, 120, 200), None, 11.0),
            ("SST", (70, 180, 100), 18.0, 0.0),
            ("A*", (150, 80, 200), 15.5, 0.0),
        ],
        race_time=20.0,
    )
    assert len(sections) == 4  # header + 3 racers
    header_lines, _ = sections[0]
    assert header_lines[0] == "STANDINGS"
    assert "1  A*" in sections[1][0][0]
    assert "2  SST" in sections[2][0][0]
    assert "3  RRT*" in sections[3][0][0]
    assert any("GOAL 15.5" in line for line in sections[1][0])
    assert any("11.0 m/s" in line for line in sections[3][0])
