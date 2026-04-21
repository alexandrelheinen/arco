"""Tests for draw_trace and TraceStyle."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from arco.simulator.viewer.trace import TraceStyle, draw_trace

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXECUTED_2D: list[tuple[float, float, float]] = [
    (0.0, 0.0, 0.0),
    (0.5, 0.1, 0.2),
    (1.0, 0.2, 0.3),
    (1.5, 0.4, 0.4),
    (2.0, 0.6, 0.5),
]

_EXECUTED_3D: list[tuple[float, float, float]] = [
    (0.0, 0.0, 0.0),
    (0.3, 0.3, 0.5),
    (0.7, 0.7, 1.0),
    (1.0, 1.0, 1.5),
]


# ---------------------------------------------------------------------------
# TraceStyle
# ---------------------------------------------------------------------------


def test_trace_style_defaults() -> None:
    style = TraceStyle()
    assert style.tail_color == "#2196f3"
    assert style.tail_alpha_start == pytest.approx(0.10)
    assert style.tail_alpha_end == pytest.approx(0.85)
    assert style.tail_length is None
    assert style.actor_color == "#f44336"
    assert style.actor_marker == "o"
    assert style.arrow_color is not None
    assert style.arrow_length > 0.0


def test_trace_style_custom() -> None:
    style = TraceStyle(
        tail_color="#00ff00",
        tail_length=3,
        actor_markersize=15.0,
        arrow_color=None,
    )
    assert style.tail_color == "#00ff00"
    assert style.tail_length == 3
    assert style.actor_markersize == pytest.approx(15.0)
    assert style.arrow_color is None


# ---------------------------------------------------------------------------
# draw_trace — 2-D
# ---------------------------------------------------------------------------


def test_draw_trace_does_not_crash_full() -> None:
    fig, ax = plt.subplots()
    draw_trace(ax, _EXECUTED_2D)
    plt.close(fig)


def test_draw_trace_raises_on_empty() -> None:
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        draw_trace(ax, [])
    plt.close(fig)


def test_draw_trace_adds_artists() -> None:
    fig, ax = plt.subplots()
    draw_trace(ax, _EXECUTED_2D)
    artists = ax.lines + list(ax.collections)
    assert len(artists) > 0
    plt.close(fig)


def test_draw_trace_step_0_only_actor() -> None:
    """At step=0 the tail is just one point so no segments are drawn."""
    fig, ax = plt.subplots()
    draw_trace(ax, _EXECUTED_2D, step=0)
    # Only the actor marker line (no tail segments)
    assert len(ax.lines) == 1  # just the actor
    plt.close(fig)


def test_draw_trace_step_last_draws_full_tail() -> None:
    fig, ax = plt.subplots()
    n = len(_EXECUTED_2D)
    draw_trace(ax, _EXECUTED_2D, step=n - 1)
    # n-1 tail segments + 1 actor + possibly 1 arrow annotation = many lines
    assert len(ax.lines) >= n - 1
    plt.close(fig)


def test_draw_trace_tail_length_limits_segments() -> None:
    fig, ax_full = plt.subplots()
    fig2, ax_short = plt.subplots()
    style_short = TraceStyle(tail_length=2, arrow_color=None)
    style_full = TraceStyle(arrow_color=None)
    draw_trace(ax_full, _EXECUTED_2D, style=style_full)
    draw_trace(ax_short, _EXECUTED_2D, style=style_short)
    # Short tail should have fewer line segments.
    assert len(ax_short.lines) < len(ax_full.lines)
    plt.close(fig)
    plt.close(fig2)


def test_draw_trace_no_arrow_when_disabled() -> None:
    fig, ax = plt.subplots()
    style = TraceStyle(arrow_color=None)
    draw_trace(ax, _EXECUTED_2D, style=style, heading_index=2)
    # No annotations should be present.
    assert len(ax.patches) == 0
    plt.close(fig)


def test_draw_trace_no_arrow_when_heading_index_none() -> None:
    fig, ax = plt.subplots()
    draw_trace(ax, _EXECUTED_2D, heading_index=None)
    assert len(ax.patches) == 0
    plt.close(fig)


def test_draw_trace_step_clamped_to_valid_range() -> None:
    fig, ax = plt.subplots()
    draw_trace(ax, _EXECUTED_2D, step=9999)
    plt.close(fig)


def test_draw_trace_single_point_executed() -> None:
    fig, ax = plt.subplots()
    draw_trace(ax, [(0.0, 0.0, 0.0)])
    # No tail segments; only actor marker.
    assert len(ax.lines) == 1
    plt.close(fig)


# ---------------------------------------------------------------------------
# draw_trace — 3-D
# ---------------------------------------------------------------------------


def test_draw_trace_3d_does_not_crash() -> None:
    fig = plt.figure()
    ax_3d = fig.add_subplot(111, projection="3d")
    draw_trace(ax_3d, _EXECUTED_3D, is_3d=True, heading_index=None)
    plt.close(fig)


def test_draw_trace_3d_adds_scatter_collection() -> None:
    fig = plt.figure()
    ax_3d = fig.add_subplot(111, projection="3d")
    draw_trace(ax_3d, _EXECUTED_3D, is_3d=True, heading_index=None)
    # At minimum the actor scatter should be present.
    assert len(ax_3d.collections) >= 1
    plt.close(fig)


def test_draw_trace_3d_no_arrow() -> None:
    """Heading arrows are 2-D only; 3-D mode must not create annotations."""
    fig = plt.figure()
    ax_3d = fig.add_subplot(111, projection="3d")
    style = TraceStyle(arrow_color="#ff0000")
    draw_trace(ax_3d, _EXECUTED_3D, style=style, is_3d=True)
    assert len(ax_3d.patches) == 0
    plt.close(fig)


# ---------------------------------------------------------------------------
# Importable from viewer
# ---------------------------------------------------------------------------


def test_draw_trace_importable_from_viewer() -> None:
    from arco.simulator.viewer import TraceStyle as TS
    from arco.simulator.viewer import draw_trace as dt

    assert dt is draw_trace
    assert TS is TraceStyle
