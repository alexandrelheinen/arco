"""Tests for FrameRenderer and LayerStyle."""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from arco.simulator.viewer.frame_renderer import FrameRenderer, LayerStyle
from arco.simulator.viewer.scene_snapshot import SceneSnapshot

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_snap(planner: str = "rrt") -> SceneSnapshot:
    """Return a minimal but complete SceneSnapshot for testing."""
    return SceneSnapshot.from_planning_result(
        scenario="test",
        planner=planner,
        start=[0.0, 0.0],
        goal=[1.0, 1.0],
        obstacles=[[0.3, 0.3], [0.4, 0.4], [0.5, 0.5]],
        tree_nodes=[[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]],
        tree_parent=[-1, 0, 1],
        found_path=[[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]],
        pruned_path=[[0.0, 0.0], [1.0, 1.0]],
        adjusted_trajectory=[[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]],
        executed_trajectory=[[0.0, 0.0], [0.4, 0.6], [1.0, 1.0]],
        metrics={"time": 0.5},
    )


def _make_snap_3d(planner: str = "rrt") -> SceneSnapshot:
    return SceneSnapshot.from_planning_result(
        scenario="ppp",
        planner=planner,
        start=[0.0, 0.0, 0.0],
        goal=[1.0, 1.0, 1.0],
        found_path=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]],
        adjusted_trajectory=[
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0],
        ],
        executed_trajectory=[
            [0.0, 0.0, 0.0],
            [0.4, 0.6, 0.4],
            [1.0, 1.0, 1.0],
        ],
        tree_nodes=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        tree_parent=[-1, 0],
    )


# ---------------------------------------------------------------------------
# LayerStyle
# ---------------------------------------------------------------------------


def test_layer_style_defaults() -> None:
    style = LayerStyle()
    assert style.color is None
    assert style.linewidth is None
    assert style.alpha is None
    assert style.markersize is None
    assert style.visible is True


def test_layer_style_visible_false_skips_layer() -> None:
    style = LayerStyle(visible=False)
    assert not style.visible


# ---------------------------------------------------------------------------
# FrameRenderer — basic 2-D render
# ---------------------------------------------------------------------------


def test_render_adds_artists_to_axes() -> None:
    fig, ax = plt.subplots()
    snap = _make_snap()
    renderer = FrameRenderer()
    renderer.render(ax, snap)
    # Any rendering should have added lines/scatter collections to the axes.
    artists = ax.lines + list(ax.collections)
    assert len(artists) > 0
    plt.close(fig)


def test_render_empty_snapshot_does_not_crash() -> None:
    fig, ax = plt.subplots()
    snap = SceneSnapshot()
    renderer = FrameRenderer()
    renderer.render(ax, snap)
    plt.close(fig)


def test_render_found_path_only() -> None:
    fig, ax = plt.subplots()
    snap = SceneSnapshot(
        planner="rrt",
        found_path=[[0.0, 0.0], [1.0, 0.0]],
    )
    renderer = FrameRenderer(
        draw_obstacles=False,
        draw_tree=False,
        draw_pruned_path=False,
        draw_trajectory=False,
        draw_executed=False,
        draw_start_goal=False,
    )
    renderer.render(ax, snap)
    assert len(ax.lines) == 1
    plt.close(fig)


def test_render_with_sst_planner_key() -> None:
    fig, ax = plt.subplots()
    snap = _make_snap(planner="sst")
    FrameRenderer().render(ax, snap)
    plt.close(fig)


# ---------------------------------------------------------------------------
# FrameRenderer — layer disable flags
# ---------------------------------------------------------------------------


def test_disable_obstacles() -> None:
    fig, ax = plt.subplots()
    snap = _make_snap()
    FrameRenderer(
        draw_obstacles=False,
        draw_tree=False,
        draw_found_path=False,
        draw_pruned_path=False,
        draw_trajectory=False,
        draw_executed=False,
        draw_start_goal=False,
    ).render(ax, snap)
    assert len(ax.lines) == 0
    assert len(ax.collections) == 0
    plt.close(fig)


def test_disable_tree_skips_tree_lines() -> None:
    fig, (ax_with, ax_without) = plt.subplots(1, 2)

    common = dict(
        draw_obstacles=False,
        draw_found_path=False,
        draw_pruned_path=False,
        draw_trajectory=False,
        draw_executed=False,
        draw_start_goal=False,
    )
    snap = _make_snap()
    FrameRenderer(draw_tree=True, **common).render(ax_with, snap)  # type: ignore[arg-type]
    FrameRenderer(draw_tree=False, **common).render(ax_without, snap)  # type: ignore[arg-type]

    assert len(ax_with.lines) > 0
    assert len(ax_without.lines) == 0
    plt.close(fig)


def test_draw_start_goal_adds_two_markers() -> None:
    fig, ax = plt.subplots()
    snap = SceneSnapshot(
        start=[0.0, 0.0],
        goal=[1.0, 1.0],
        planner="rrt",
    )
    FrameRenderer(
        draw_obstacles=False,
        draw_tree=False,
        draw_found_path=False,
        draw_pruned_path=False,
        draw_trajectory=False,
        draw_executed=False,
        draw_start_goal=True,
    ).render(ax, snap)
    assert len(ax.lines) == 2  # start "s" + goal "x"
    plt.close(fig)


# ---------------------------------------------------------------------------
# FrameRenderer — 3-D rendering
# ---------------------------------------------------------------------------


def test_render_3d_does_not_crash() -> None:
    fig = plt.figure()
    ax_3d = fig.add_subplot(111, projection="3d")
    snap = _make_snap_3d()
    renderer = FrameRenderer(is_3d=True)
    renderer.render(ax_3d, snap)
    plt.close(fig)


def test_render_3d_start_goal_markers() -> None:
    fig = plt.figure()
    ax_3d = fig.add_subplot(111, projection="3d")
    snap = SceneSnapshot(
        start=[0.0, 0.0, 0.0],
        goal=[1.0, 1.0, 1.0],
        planner="rrt",
    )
    FrameRenderer(
        draw_obstacles=False,
        draw_tree=False,
        draw_found_path=False,
        draw_pruned_path=False,
        draw_trajectory=False,
        draw_executed=False,
        draw_start_goal=True,
        is_3d=True,
    ).render(ax_3d, snap)
    # 3-D start/goal use scatter collections, not ax.lines.
    assert len(ax_3d.collections) == 2
    plt.close(fig)


# ---------------------------------------------------------------------------
# FrameRenderer — style overrides
# ---------------------------------------------------------------------------


def test_layer_style_override_changes_color() -> None:
    fig, ax = plt.subplots()
    snap = SceneSnapshot(
        planner="rrt",
        found_path=[[0.0, 0.0], [1.0, 0.0]],
    )
    renderer = FrameRenderer(
        draw_obstacles=False,
        draw_tree=False,
        draw_pruned_path=False,
        draw_trajectory=False,
        draw_executed=False,
        draw_start_goal=False,
        styles={"found_path": LayerStyle(color="#ff0000")},
    )
    renderer.render(ax, snap)
    assert len(ax.lines) == 1
    line_color = ax.lines[0].get_color()
    assert line_color == "#ff0000"
    plt.close(fig)


def test_layer_style_visible_false_hides_found_path() -> None:
    fig, ax = plt.subplots()
    snap = SceneSnapshot(
        planner="rrt",
        found_path=[[0.0, 0.0], [1.0, 0.0]],
    )
    renderer = FrameRenderer(
        draw_obstacles=False,
        draw_tree=False,
        draw_pruned_path=False,
        draw_trajectory=False,
        draw_executed=False,
        draw_start_goal=False,
        styles={"found_path": LayerStyle(visible=False)},
    )
    renderer.render(ax, snap)
    assert len(ax.lines) == 0
    plt.close(fig)


# ---------------------------------------------------------------------------
# FrameRenderer — importable from viewer
# ---------------------------------------------------------------------------


def test_frame_renderer_importable_from_viewer() -> None:
    from arco.simulator.viewer import FrameRenderer as FR
    from arco.simulator.viewer import LayerStyle as LS

    assert FR is FrameRenderer
    assert LS is LayerStyle


# ---------------------------------------------------------------------------
# FrameRenderer — pruned-path glow squares
# ---------------------------------------------------------------------------


def test_pruned_path_renders_as_square_markers() -> None:
    """Pruned landmarks must be scatter collections (squares, no lines)."""
    fig, ax = plt.subplots()
    snap = SceneSnapshot(
        planner="rrt",
        pruned_path=[[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]],
    )
    renderer = FrameRenderer(
        draw_obstacles=False,
        draw_tree=False,
        draw_found_path=False,
        draw_trajectory=False,
        draw_executed=False,
        draw_start_goal=False,
        draw_pruned_path=True,
    )
    renderer.render(ax, snap)
    # No connecting lines — only a single scatter collection of squares.
    assert len(ax.lines) == 0
    assert len(ax.collections) == 1
    plt.close(fig)


def test_pruned_path_visible_false_skips_glow() -> None:
    fig, ax = plt.subplots()
    snap = SceneSnapshot(
        planner="rrt",
        pruned_path=[[0.0, 0.0], [1.0, 1.0]],
    )
    renderer = FrameRenderer(
        draw_obstacles=False,
        draw_tree=False,
        draw_found_path=False,
        draw_trajectory=False,
        draw_executed=False,
        draw_start_goal=False,
        draw_pruned_path=True,
        styles={"pruned_path": LayerStyle(visible=False)},
    )
    renderer.render(ax, snap)
    assert len(ax.lines) == 0
    assert len(ax.collections) == 0
    plt.close(fig)
