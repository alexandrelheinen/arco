"""Tests for the shared StandardLayout and viewer utility functions."""

from __future__ import annotations

import os
import sys

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

# Ensure arco is importable when running from the repo root without install.
_REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, os.path.join(_REPO, "src"))

from arco.tools.viewer.layout import StandardLayout
from arco.tools.viewer.utils import format_clock, polyline_length

# ---------------------------------------------------------------------------
# polyline_length
# ---------------------------------------------------------------------------


def test_polyline_length_none_returns_zero() -> None:
    assert polyline_length(None) == 0.0


def test_polyline_length_empty_returns_zero() -> None:
    assert polyline_length([]) == 0.0


def test_polyline_length_single_point_returns_zero() -> None:
    assert polyline_length([np.array([1.0, 2.0])]) == 0.0


def test_polyline_length_two_points_horizontal() -> None:
    path = [np.array([0.0, 0.0]), np.array([3.0, 0.0])]
    assert polyline_length(path) == pytest.approx(3.0)


def test_polyline_length_three_points() -> None:
    path = [
        np.array([0.0, 0.0]),
        np.array([3.0, 0.0]),
        np.array([3.0, 4.0]),
    ]
    assert polyline_length(path) == pytest.approx(7.0)


def test_polyline_length_3d_path() -> None:
    path = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])]
    assert polyline_length(path) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# format_clock
# ---------------------------------------------------------------------------


def test_format_clock_zero() -> None:
    assert format_clock(0.0) == "00min00s"


def test_format_clock_seconds_only() -> None:
    assert format_clock(7.0) == "00min07s"


def test_format_clock_rounds_to_nearest() -> None:
    assert format_clock(7.4) == "00min07s"
    assert format_clock(7.6) == "00min08s"


def test_format_clock_one_minute() -> None:
    assert format_clock(60.0) == "01min00s"


def test_format_clock_mixed() -> None:
    assert format_clock(125.0) == "02min05s"


def test_format_clock_negative_treated_as_zero() -> None:
    assert format_clock(-5.0) == "00min00s"


# ---------------------------------------------------------------------------
# StandardLayout.create
# ---------------------------------------------------------------------------


def test_create_returns_four_tuple() -> None:
    result = StandardLayout.create()
    assert len(result) == 4


def test_create_default_figsize() -> None:
    fig, ax_ws, ax_cs, ax_bottom = StandardLayout.create()
    w, h = fig.get_size_inches()
    assert (w, h) == pytest.approx(StandardLayout.FIGSIZE)
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_create_custom_figsize() -> None:
    fig, _, _, _ = StandardLayout.create(figsize=(10, 6))
    w, h = fig.get_size_inches()
    assert (w, h) == pytest.approx((10, 6))
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_create_has_suptitle_when_provided() -> None:
    fig, _, _, _ = StandardLayout.create(title="Test Title")
    # suptitle stores the text object
    assert fig._suptitle is not None  # type: ignore[attr-defined]
    assert fig._suptitle.get_text() == "Test Title"  # type: ignore[attr-defined]
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_create_no_suptitle_when_empty() -> None:
    fig, _, _, _ = StandardLayout.create()
    assert fig._suptitle is None  # type: ignore[attr-defined]
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_create_bottom_axis_off() -> None:
    fig, _, _, ax_bottom = StandardLayout.create()
    assert not ax_bottom.axison
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_create_2d_by_default() -> None:
    fig, ax_ws, ax_cs, _ = StandardLayout.create()
    # 2-D axes do NOT have a 'get_zlim' attribute.
    assert not hasattr(ax_ws, "get_zlim")
    assert not hasattr(ax_cs, "get_zlim")
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_create_ws_3d_flag() -> None:
    fig, ax_ws, ax_cs, _ = StandardLayout.create(ws_3d=True)
    assert hasattr(ax_ws, "get_zlim")
    assert not hasattr(ax_cs, "get_zlim")
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_create_cs_3d_flag() -> None:
    fig, ax_ws, ax_cs, _ = StandardLayout.create(cs_3d=True)
    assert not hasattr(ax_ws, "get_zlim")
    assert hasattr(ax_cs, "get_zlim")
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_create_both_3d() -> None:
    fig, ax_ws, ax_cs, _ = StandardLayout.create(ws_3d=True, cs_3d=True)
    assert hasattr(ax_ws, "get_zlim")
    assert hasattr(ax_cs, "get_zlim")
    import matplotlib.pyplot as plt

    plt.close(fig)


# ---------------------------------------------------------------------------
# StandardLayout.write_metrics
# ---------------------------------------------------------------------------


def test_write_metrics_adds_text_to_bottom() -> None:
    fig, _, _, ax_bottom = StandardLayout.create()
    StandardLayout.write_metrics(ax_bottom, ["line1", "line2"])
    texts = ax_bottom.texts
    assert len(texts) == 1
    assert "line1" in texts[0].get_text()
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_write_metrics_empty_lines_no_text() -> None:
    fig, _, _, ax_bottom = StandardLayout.create()
    StandardLayout.write_metrics(ax_bottom, [])
    assert len(ax_bottom.texts) == 0
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_write_metrics_columns_splits_lines() -> None:
    fig, _, _, ax_bottom = StandardLayout.create()
    StandardLayout.write_metrics(ax_bottom, ["a", "b", "c", "d"], columns=2)
    # 4 lines split across 2 columns → 2 text objects
    assert len(ax_bottom.texts) == 2
    import matplotlib.pyplot as plt

    plt.close(fig)


# ---------------------------------------------------------------------------
# StandardLayout — viewer __init__ re-exports
# ---------------------------------------------------------------------------


def test_standard_layout_importable_from_viewer() -> None:
    from arco.tools.viewer import StandardLayout as SL  # noqa: F401

    assert SL is StandardLayout


def test_utils_importable_from_viewer() -> None:
    from arco.tools.viewer import format_clock as fc  # noqa: F401
    from arco.tools.viewer import polyline_length as pl  # noqa: F401

    assert pl is polyline_length
    assert fc is format_clock
