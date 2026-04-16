"""Tests for the ScreenLayout geometry dataclass."""

from __future__ import annotations

from arco.tools.simulator.sim.layout import ScreenLayout


def test_content_dimensions() -> None:
    layout = ScreenLayout(sw=1280, sh=720)
    assert layout.content_x == 260
    assert layout.content_y == 30
    assert layout.content_w == 1020
    assert layout.content_h == 650


def test_content_dimensions_custom() -> None:
    layout = ScreenLayout(
        sw=800, sh=600, sidebar_w=200, header_h=50, footer_h=25
    )
    assert layout.content_w == 600
    assert layout.content_h == 525


def test_content_w_minimum_one() -> None:
    layout = ScreenLayout(sw=100, sh=100, sidebar_w=200)
    assert layout.content_w == 1


def test_content_h_minimum_one() -> None:
    layout = ScreenLayout(sw=100, sh=50, header_h=30, footer_h=25)
    assert layout.content_h == 1


def test_default_sidebar_w() -> None:
    layout = ScreenLayout(sw=1280, sh=720)
    assert layout.sidebar_w == 260


def test_default_header_footer() -> None:
    layout = ScreenLayout(sw=1280, sh=720)
    assert layout.header_h == 40
    assert layout.footer_h == 30


def test_content_x_equals_sidebar_w() -> None:
    layout = ScreenLayout(sw=1280, sh=720, sidebar_w=300)
    assert layout.content_x == 300


def test_content_y_equals_footer_h() -> None:
    layout = ScreenLayout(sw=1280, sh=720, footer_h=45)
    assert layout.content_y == 45
