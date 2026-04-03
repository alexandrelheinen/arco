"""Tests for tools/graph/generator.py."""

from __future__ import annotations

import math
import os
import sys

import pytest

pytest.importorskip("scipy")

# Expose arco package and tools/ config loader.
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "..", "src"),
)
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "..", "tools"),
)

from graph.generator import generate_graph

# ---------------------------------------------------------------------------
# Shared fixture config
# ---------------------------------------------------------------------------

_BASE_CFG = {
    "type": "ring",
    "width": 400,
    "height": 400,
    "mean_edge_length": 60,
    "hole_count": 2,
    "hole_radius": 45,
    "curvature": 0.09,
    "seed": 7,
}


def _cfg(**overrides):
    return {**_BASE_CFG, **overrides}


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_seeded_output_is_deterministic():
    g1 = generate_graph(_cfg())
    g2 = generate_graph(_cfg())
    assert [n for n in g1.nodes] == [n for n in g2.nodes]
    assert len(g1.edges) == len(g2.edges)


def test_different_seeds_differ():
    g1 = generate_graph(_cfg(seed=1))
    g2 = generate_graph(_cfg(seed=2))
    # Very unlikely that two differently-seeded meshes have the same edge count.
    assert len(g1.nodes) != len(g2.nodes) or len(g1.edges) != len(g2.edges)


# ---------------------------------------------------------------------------
# Spatial constraints
# ---------------------------------------------------------------------------


def test_nodes_inside_bounding_box():
    cfg = _cfg()
    g = generate_graph(cfg)
    for n in g.nodes:
        x, y = g.position(n)
        assert 0.0 <= x <= cfg["width"]
        assert 0.0 <= y <= cfg["height"]


def test_no_nodes_inside_hole_radius():
    cfg = _cfg(hole_count=1)
    g = generate_graph(cfg)
    cx, cy = cfg["width"] / 2, cfg["height"] / 2
    hole_radius = cfg["hole_radius"]
    for n in g.nodes:
        x, y = g.position(n)
        assert math.hypot(x - cx, y - cy) >= hole_radius


# ---------------------------------------------------------------------------
# Edge constraints
# ---------------------------------------------------------------------------


def test_all_edges_within_max_length():
    cfg = _cfg()
    g = generate_graph(cfg)
    max_length = 1.7 * cfg["mean_edge_length"]
    for a, b, _ in g.edges:
        ax, ay = g.position(a)
        bx, by = g.position(b)
        assert math.hypot(ax - bx, ay - by) <= max_length + 1e-9


def test_each_edge_has_two_waypoints():
    g = generate_graph(_cfg())
    for a, b, _ in g.edges:
        wps = g.edge_geometry(a, b)
        assert len(wps) == 2


def test_waypoints_progress_along_chord():
    """Both waypoints must have projection t in (0, 1) along the chord AB."""
    g = generate_graph(_cfg())
    for a, b, _ in g.edges:
        ax, ay = g.position(a)
        bx, by = g.position(b)
        dx, dy = bx - ax, by - ay
        chord_sq = dx * dx + dy * dy
        for wx, wy in g.edge_geometry(a, b):
            t = ((wx - ax) * dx + (wy - ay) * dy) / chord_sq
            assert 0.0 < t < 1.0


# ---------------------------------------------------------------------------
# Hole count variants
# ---------------------------------------------------------------------------


def test_single_central_hole():
    g = generate_graph(_cfg(hole_count=1))
    assert len(g.nodes) > 0
    assert len(g.edges) > 0


def test_multiple_holes_reduce_node_count():
    g_few = generate_graph(_cfg(hole_count=1, seed=99))
    g_many = generate_graph(_cfg(hole_count=4, seed=99))
    assert len(g_few.nodes) > len(g_many.nodes)


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------


def test_larger_area_produces_more_nodes():
    g_small = generate_graph(_cfg(width=200, height=200, seed=5))
    g_large = generate_graph(_cfg(width=600, height=600, seed=5))
    assert len(g_large.nodes) > len(g_small.nodes)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown graph type"):
        generate_graph({**_BASE_CFG, "type": "invalid_type"})


def test_missing_required_key_raises():
    cfg = dict(_BASE_CFG)
    del cfg["mean_edge_length"]
    with pytest.raises(KeyError):
        generate_graph(cfg)
