"""Tests for the road graph JSON loader."""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from arco.mapping.graph.loader import load_road_graph
from arco.mapping.graph.road import RoadGraph

_CITY_NETWORK = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "tools",
    "config",
    "city_network.json",
)


class TestLoadRoadGraphCity:
    def test_returns_road_graph(self):
        graph = load_road_graph(_CITY_NETWORK)
        assert isinstance(graph, RoadGraph)

    def test_node_count(self):
        graph = load_road_graph(_CITY_NETWORK)
        assert len(graph.nodes) == 20

    def test_edge_count(self):
        graph = load_road_graph(_CITY_NETWORK)
        assert len(graph.edges) == 40

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

    def test_inner_north_node_position(self):
        """Node 0 is the inner-North node at (300, 410)."""
        graph = load_road_graph(_CITY_NETWORK)
        pos = graph.position(0)
        assert abs(pos[0] - 300.0) < 1.0
        assert abs(pos[1] - 410.0) < 1.0


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
