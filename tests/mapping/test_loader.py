"""Tests for the road graph JSON loader."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Optional

import numpy as np
import pytest

from arco.mapping.graph.loader import load_road_graph
from arco.mapping.graph.road import RoadGraph


def _resolve_city_network_path() -> Optional[str]:
    """Return the existing city-network descriptor path.

    Supports both the legacy ``city_network.json`` and the newer
    ``city.json`` filenames.
    """
    config_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "src",
        "arco",
        "tools",
        "map",
    )
    for filename in ("city_network.json", "city.json"):
        path = os.path.join(config_dir, filename)
        if os.path.isfile(path):
            return path
    return None


_CITY_NETWORK = _resolve_city_network_path()


pytestmark = pytest.mark.skipif(
    _CITY_NETWORK is None,
    reason=(
        "Legacy city network descriptor was removed; "
        "city_network.json/city.json not present in src/arco/tools/map/."
    ),
)


class TestLoadRoadGraphCity:
    def test_returns_road_graph(self):
        graph = load_road_graph(_CITY_NETWORK)
        assert isinstance(graph, RoadGraph)

    def test_node_count(self):
        graph = load_road_graph(_CITY_NETWORK)
        assert len(graph.nodes) == 61

    def test_edge_count(self):
        graph = load_road_graph(_CITY_NETWORK)
        assert len(graph.edges) == 110

    def test_all_edges_have_waypoints(self):
        graph = load_road_graph(_CITY_NETWORK)
        for a, b, _ in graph.edges:
            pts = graph.edge_geometry(a, b)
            assert len(pts) >= 1, f"Edge ({a},{b}) has no waypoints"

    def test_node_positions_are_finite(self):
        graph = load_road_graph(_CITY_NETWORK)
        for nid in graph.nodes:
            pos = graph.position(nid)
            assert np.all(np.isfinite(pos))

    def test_terminal_north_node_position(self):
        """Terminal N node (id=57) should be at (365, 1070)."""
        graph = load_road_graph(_CITY_NETWORK)
        pos = graph.position(57)
        assert abs(pos[0] - 365.0) < 1.0
        assert abs(pos[1] - 1070.0) < 1.0

    def test_terminal_south_node_position(self):
        """Terminal S node (id=59) should be at (365, -70)."""
        graph = load_road_graph(_CITY_NETWORK)
        pos = graph.position(59)
        assert abs(pos[0] - 365.0) < 1.0
        assert abs(pos[1] - (-70.0)) < 1.0

    def test_waypoints_no_back_and_forth(self):
        """All waypoint t-values must lie in (0, 1) for canonical direction."""
        graph = load_road_graph(_CITY_NETWORK)
        for a, b, _ in graph.edges:
            xa, ya = graph.position(a)
            xb, yb = graph.position(b)
            dx, dy = xb - xa, yb - ya
            lsq = dx * dx + dy * dy
            if lsq < 1e-9:
                continue
            for wx, wy in graph.edge_geometry(a, b):
                t = ((wx - xa) * dx + (wy - ya) * dy) / lsq
                assert (
                    0.0 < t < 1.0
                ), f"Edge ({a},{b}) waypoint ({wx},{wy}) has t={t:.3f}"


class TestLoadRoadGraphErrors:
    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_road_graph("/nonexistent/path/to/network.json")

    def test_unknown_node_in_edge_raises(self):
        bad_data = {
            "nodes": [{"id": 0, "x": 0.0, "y": 0.0}],
            "edges": [{"from": 0, "to": 99, "waypoints": []}],
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as fh:
            json.dump(bad_data, fh)
            path = fh.name
        try:
            with pytest.raises(ValueError, match="99"):
                load_road_graph(path)
        finally:
            os.unlink(path)

    def test_minimal_valid_network(self):
        data = {
            "nodes": [
                {"id": 0, "x": 0.0, "y": 0.0},
                {"id": 1, "x": 10.0, "y": 5.0},
            ],
            "edges": [
                {
                    "from": 0,
                    "to": 1,
                    "waypoints": [[5.0, 1.0], [7.0, 3.0]],
                }
            ],
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as fh:
            json.dump(data, fh)
            path = fh.name
        try:
            graph = load_road_graph(path)
            assert len(graph.nodes) == 2
            assert len(graph.edges) == 1
            wps = graph.edge_geometry(0, 1)
            assert len(wps) == 2
            assert wps[0] == (5.0, 1.0)
        finally:
            os.unlink(path)

    def test_no_waypoints_loads_straight_edge(self):
        data = {
            "nodes": [
                {"id": 0, "x": 0.0, "y": 0.0},
                {"id": 1, "x": 10.0, "y": 0.0},
            ],
            "edges": [{"from": 0, "to": 1}],
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as fh:
            json.dump(data, fh)
            path = fh.name
        try:
            graph = load_road_graph(path)
            assert len(graph.edges) == 1
            assert graph.edge_geometry(0, 1) == []
        finally:
            os.unlink(path)

    def test_reverse_edge_waypoints_are_normalized(self):
        """Loader must reverse waypoints when from > to (canonical direction)."""
        data = {
            "nodes": [
                {"id": 0, "x": 0.0, "y": 0.0},
                {"id": 5, "x": 10.0, "y": 0.0},
            ],
            # Edge defined "backwards": from=5 > to=0
            # Waypoints designed for 5→0: wp1 at x=7, wp2 at x=3
            "edges": [
                {
                    "from": 5,
                    "to": 0,
                    "waypoints": [[7.0, 0.5], [3.0, -0.5]],
                }
            ],
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as fh:
            json.dump(data, fh)
            path = fh.name
        try:
            graph = load_road_graph(path)
            # Waypoints defined for 5→0: [7.0, 0.5] then [3.0, -0.5].
            # Loader reverses them to canonical 0→5 order: [(3.0, -0.5), (7.0, 0.5)].
            wps = graph.edge_geometry(0, 5)
            assert wps == [(3.0, -0.5), (7.0, 0.5)]
        finally:
            os.unlink(path)
