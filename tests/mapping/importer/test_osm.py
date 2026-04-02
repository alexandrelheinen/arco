"""Unit tests for OSMImporter.

All tests mock the osmnx, networkx, and pyproj libraries so that no network
access or optional dependencies are required.  Each test exercises a specific
aspect of the conversion logic.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest

from arco.mapping.importer.osm import OSMImporter, _METRES_PER_DEGREE


# ---------------------------------------------------------------------------
# Lightweight fake graph that mimics a networkx MultiDiGraph as returned by
# osmnx after projection.  Using a real class (rather than MagicMock) lets
# tests control iteration precisely without needing networkx installed.
# ---------------------------------------------------------------------------


class _FakeGraph:
    """Minimal stand-in for a projected osmnx / networkx MultiDiGraph."""

    def __init__(self, crs: str = "EPSG:32631") -> None:
        self._nodes: Dict[int, Dict[str, float]] = {}
        self._edges: List[Tuple[int, int, Dict[str, Any]]] = []
        self.graph: Dict[str, str] = {"crs": crs}

    def add_node(self, osmid: int, x: float, y: float) -> None:
        """Add a node with projected coordinates."""
        self._nodes[osmid] = {"x": x, "y": y}

    def add_edge(
        self,
        u: int,
        v: int,
        length: float = 100.0,
        geometry: Optional[Any] = None,
    ) -> None:
        """Add a directed edge with optional Shapely geometry."""
        data: Dict[str, Any] = {"length": length}
        if geometry is not None:
            data["geometry"] = geometry
        self._edges.append((u, v, data))

    def nodes(self, data: bool = False):
        """Return node view compatible with networkx iteration."""
        if data:
            return list(self._nodes.items())
        return list(self._nodes.keys())

    def edges(self, data: bool = False):
        """Return edge view compatible with networkx iteration."""
        if data:
            return list(self._edges)
        return [(u, v) for u, v, _ in self._edges]

    def number_of_nodes(self) -> int:
        """Return node count."""
        return len(self._nodes)

    def number_of_edges(self) -> int:
        """Return edge count."""
        return len(self._edges)

    def subgraph(self, nodes) -> "_FakeGraph":
        """Return a view restricted to *nodes*."""
        node_set = set(nodes)
        sub = _FakeGraph(self.graph["crs"])
        for osmid, d in self._nodes.items():
            if osmid in node_set:
                sub.add_node(osmid, d["x"], d["y"])
        for u, v, d in self._edges:
            if u in node_set and v in node_set:
                sub.add_edge(u, v, d.get("length", 100.0), d.get("geometry"))
        return sub

    def copy(self) -> "_FakeGraph":
        """Return a shallow copy."""
        clone = _FakeGraph(self.graph["crs"])
        clone._nodes = dict(self._nodes)
        clone._edges = list(self._edges)
        return clone


def _five_node_graph() -> _FakeGraph:
    """Return a simple 5-node chain graph (fully connected)."""
    G = _FakeGraph()
    for i in range(5):
        G.add_node(1000 + i, x=float(i * 100), y=float(i * 50))
    for i in range(4):
        G.add_edge(1000 + i, 1001 + i, length=111.0)
    return G


def _make_mock_ox(
    raw_graph: _FakeGraph,
    proj_graph: Optional[_FakeGraph] = None,
) -> MagicMock:
    """Return a MagicMock configured as a minimal osmnx module."""
    if proj_graph is None:
        proj_graph = raw_graph
    mock_ox = MagicMock()
    mock_ox.graph_from_bbox.return_value = raw_graph
    mock_ox.project_graph.return_value = proj_graph
    return mock_ox


# ---------------------------------------------------------------------------
# Tests: bounding-box computation (pure math, no osmnx needed)
# ---------------------------------------------------------------------------


class TestBboxComputation:
    def test_symmetric_margin(self):
        min_lat, max_lat, min_lon, max_lon = OSMImporter._compute_bbox(
            lat_start=48.86,
            lon_start=2.33,
            lat_goal=48.85,
            lon_goal=2.29,
            margin_m=_METRES_PER_DEGREE,
        )
        assert min_lat < 48.85
        assert max_lat > 48.86
        assert min_lon < 2.29
        assert max_lon > 2.33

    def test_margin_zero(self):
        min_lat, max_lat, min_lon, max_lon = OSMImporter._compute_bbox(
            lat_start=1.0,
            lon_start=1.0,
            lat_goal=2.0,
            lon_goal=2.0,
            margin_m=0.0,
        )
        assert min_lat == pytest.approx(1.0)
        assert max_lat == pytest.approx(2.0)
        assert min_lon == pytest.approx(1.0)
        assert max_lon == pytest.approx(2.0)

    def test_delta_one_degree(self):
        """One degree of margin (111 km) should add ~1 degree on each side."""
        min_lat, max_lat, min_lon, max_lon = OSMImporter._compute_bbox(
            lat_start=0.0,
            lon_start=0.0,
            lat_goal=0.0,
            lon_goal=0.0,
            margin_m=_METRES_PER_DEGREE,
        )
        assert min_lat == pytest.approx(-1.0)
        assert max_lat == pytest.approx(1.0)
        assert min_lon == pytest.approx(-1.0)
        assert max_lon == pytest.approx(1.0)

    def test_bbox_order_independent(self):
        """Swapping start and goal should produce the same bbox."""
        bbox1 = OSMImporter._compute_bbox(1.0, 2.0, 3.0, 4.0, 500.0)
        bbox2 = OSMImporter._compute_bbox(3.0, 4.0, 1.0, 2.0, 500.0)
        for a, b in zip(bbox1, bbox2):
            assert a == pytest.approx(b)


# ---------------------------------------------------------------------------
# Tests: from_coords with mocked osmnx
# ---------------------------------------------------------------------------


class TestFromCoords:
    def _run(
        self,
        raw_graph: _FakeGraph,
        proj_graph: Optional[_FakeGraph] = None,
    ) -> Tuple[Any, Any, Any]:
        """Call from_coords with the given mock graphs."""
        mock_ox = _make_mock_ox(raw_graph, proj_graph)
        importer = OSMImporter()
        with patch.object(OSMImporter, "_require_osmnx", return_value=mock_ox):
            with patch.object(
                OSMImporter,
                "_project_latlon",
                return_value=(50.0, 100.0),
            ):
                with patch.object(
                    OSMImporter,
                    "_largest_component",
                    side_effect=lambda ox_mod, G: G,
                ):
                    return importer.from_coords(
                        lat_start=48.86,
                        lon_start=2.33,
                        lat_goal=48.85,
                        lon_goal=2.29,
                        margin_m=500.0,
                        network_type="bike",
                    )

    def test_node_count(self):
        graph, _, _ = self._run(_five_node_graph())
        assert len(graph.nodes) == 5

    def test_edge_count(self):
        graph, _, _ = self._run(_five_node_graph())
        assert len(graph.edges) == 4

    def test_node_ids_are_zero_based(self):
        graph, _, _ = self._run(_five_node_graph())
        assert sorted(graph.nodes) == list(range(5))

    def test_origin_shift_applied(self):
        """After shift, minimum node coordinate should be 0."""
        graph, _, _ = self._run(_five_node_graph())
        xs = [float(graph.position(nid)[0]) for nid in graph.nodes]
        ys = [float(graph.position(nid)[1]) for nid in graph.nodes]
        assert min(xs) == pytest.approx(0.0)
        assert min(ys) == pytest.approx(0.0)

    def test_start_goal_returned(self):
        _, start_xy, goal_xy = self._run(_five_node_graph())
        assert start_xy == (50.0, 100.0)
        assert goal_xy == (50.0, 100.0)

    def test_edge_weight_from_length(self):
        """Edge weight should equal the 'length' attribute from the OSM data."""
        G = _FakeGraph()
        G.add_node(100, x=0.0, y=0.0)
        G.add_node(101, x=250.0, y=0.0)
        G.add_edge(100, 101, length=250.0)

        graph, _, _ = self._run(G)
        # The single edge should have weight = 250.0
        edges = list(graph.edges)
        assert len(edges) == 1
        _, _, weight = edges[0]
        assert weight == pytest.approx(250.0)

    def test_graph_from_bbox_called(self):
        """from_coords must call ox.graph_from_bbox with a bbox tuple."""
        G = _five_node_graph()
        mock_ox = _make_mock_ox(G)
        importer = OSMImporter()
        with patch.object(OSMImporter, "_require_osmnx", return_value=mock_ox):
            with patch.object(
                OSMImporter, "_project_latlon", return_value=(0.0, 0.0)
            ):
                with patch.object(
                    OSMImporter,
                    "_largest_component",
                    side_effect=lambda ox_mod, G: G,
                ):
                    importer.from_coords(
                        48.86, 2.33, 48.85, 2.29, margin_m=500.0
                    )
        mock_ox.graph_from_bbox.assert_called_once()
        # First positional arg must be a 4-element (left, bottom, right, top)
        # bbox tuple — no longer uses north/south/east/west keyword args.
        call_args = mock_ox.graph_from_bbox.call_args
        bbox_arg = call_args[0][0]  # first positional argument
        assert len(bbox_arg) == 4

    def test_edge_geometry_waypoints_stored(self):
        """Intermediate LineString coordinates must be stored as waypoints."""

        class _FakeLineString:
            @property
            def coords(self):
                return [
                    (0.0, 0.0),
                    (50.0, 10.0),
                    (100.0, 0.0),
                ]

        G = _FakeGraph()
        G.add_node(200, x=0.0, y=0.0)
        G.add_node(201, x=100.0, y=0.0)
        G.add_edge(200, 201, length=105.0, geometry=_FakeLineString())

        graph, _, _ = self._run(G)
        # The intermediate waypoint (50, 10) must be stored (origin-shifted)
        waypoints = graph.edge_geometry(0, 1)
        assert len(waypoints) == 1
        assert waypoints[0][0] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Tests: largest-connected-component pruning
# ---------------------------------------------------------------------------


class TestLargestComponent:
    def test_lcc_uses_networkx(self):
        """_largest_component uses networkx weakly_connected_components."""
        try:
            import networkx as nx
        except ImportError:
            pytest.skip("networkx not installed")

        G_nx = nx.MultiDiGraph()
        for i in range(5):
            G_nx.add_node(i)
        for i in range(4):
            G_nx.add_edge(i, i + 1)

        mock_ox = MagicMock()
        result = OSMImporter._largest_component(mock_ox, G_nx)
        assert result.number_of_nodes() == 5

    def test_disconnected_graph_pruned_via_lcc(self):
        """The networkx path prunes smaller components."""
        try:
            import networkx as nx
        except ImportError:
            pytest.skip("networkx not installed")

        G_nx = nx.MultiDiGraph()
        for i in range(5):
            G_nx.add_node(i)
        for i in range(4):
            G_nx.add_edge(i, i + 1, length=100.0)
        # Node 5 is isolated
        G_nx.add_node(5)

        mock_ox = MagicMock()
        result = OSMImporter._largest_component(mock_ox, G_nx)
        assert 5 not in result.nodes()
        assert all(i in result.nodes() for i in range(5))

    def test_single_component_unchanged(self):
        """A fully connected graph should be returned with all nodes intact."""
        try:
            import networkx as nx
        except ImportError:
            pytest.skip("networkx not installed")

        G_nx = nx.MultiDiGraph()
        for i in range(5):
            G_nx.add_node(i)
        for i in range(4):
            G_nx.add_edge(i, i + 1)

        mock_ox = MagicMock()
        result = OSMImporter._largest_component(mock_ox, G_nx)
        assert result.number_of_nodes() == 5

    def test_lcc_removes_smaller_component(self):
        """Integration: disconnected OSM graph is pruned before RoadGraph build."""
        # Component A: nodes 1000-1004 (5 nodes, 4 edges)
        G_full = _FakeGraph()
        for i in range(5):
            G_full.add_node(1000 + i, x=float(i * 100), y=0.0)
        for i in range(4):
            G_full.add_edge(1000 + i, 1001 + i, length=100.0)
        # Component B: isolated node 9999
        G_full.add_node(9999, x=5000.0, y=5000.0)

        # Simulate LCC extraction returning only component A
        G_pruned = _FakeGraph()
        for i in range(5):
            G_pruned.add_node(1000 + i, x=float(i * 100), y=0.0)
        for i in range(4):
            G_pruned.add_edge(1000 + i, 1001 + i, length=100.0)

        mock_ox = _make_mock_ox(raw_graph=G_full, proj_graph=G_pruned)

        importer = OSMImporter()
        with patch.object(OSMImporter, "_require_osmnx", return_value=mock_ox):
            with patch.object(
                OSMImporter, "_project_latlon", return_value=(0.0, 0.0)
            ):
                # Patch _largest_component to return the pruned graph
                with patch.object(
                    OSMImporter,
                    "_largest_component",
                    return_value=G_pruned,
                ):
                    graph, _, _ = importer.from_coords(
                        48.86, 2.33, 48.85, 2.29, margin_m=500.0
                    )

        # Isolated node 9999 must not be in the final RoadGraph
        assert len(graph.nodes) == 5


# ---------------------------------------------------------------------------
# Tests: from_addresses
# ---------------------------------------------------------------------------


class TestFromAddresses:
    def test_geocoding_failure_raises_value_error(self):
        """A failed geocode must raise ValueError with a descriptive message."""
        mock_ox = MagicMock()
        mock_ox.geocode.side_effect = Exception("Nominatim returned no result")

        importer = OSMImporter()
        with patch.object(OSMImporter, "_require_osmnx", return_value=mock_ox):
            with pytest.raises(ValueError, match="Failed to geocode"):
                importer.from_addresses(
                    address_start="Nowhere, Nowhere",
                    address_goal="Tour Eiffel, Paris, France",
                )

    def test_geocoding_calls_from_coords(self):
        """from_addresses must delegate to from_coords after geocoding."""
        G = _five_node_graph()
        mock_ox = _make_mock_ox(G)
        mock_ox.geocode.side_effect = [
            (48.86, 2.33),  # start
            (48.85, 2.29),  # goal
        ]

        importer = OSMImporter()
        with patch.object(OSMImporter, "_require_osmnx", return_value=mock_ox):
            with patch.object(
                importer, "from_coords", wraps=importer.from_coords
            ) as spy:
                with patch.object(
                    OSMImporter, "_project_latlon", return_value=(0.0, 0.0)
                ):
                    with patch.object(
                        OSMImporter,
                        "_largest_component",
                        side_effect=lambda ox_mod, G: G,
                    ):
                        importer.from_addresses(
                            "Musée du Louvre, Paris",
                            "Tour Eiffel, Paris",
                        )
            spy.assert_called_once_with(
                48.86, 2.33, 48.85, 2.29, 500.0, "bike"
            )

    def test_goal_geocoding_failure_raises_value_error(self):
        """Geocoding failure on the *goal* address must also raise ValueError."""
        mock_ox = MagicMock()
        mock_ox.geocode.side_effect = [
            (48.86, 2.33),  # start succeeds
            Exception("Nominatim 404"),  # goal fails
        ]

        importer = OSMImporter()
        with patch.object(OSMImporter, "_require_osmnx", return_value=mock_ox):
            with pytest.raises(ValueError, match="Failed to geocode"):
                importer.from_addresses(
                    "Musée du Louvre, Paris, France",
                    "Nulle Part, Nulle Part",
                )


# ---------------------------------------------------------------------------
# Tests: ImportError when osmnx is missing
# ---------------------------------------------------------------------------


class TestImportError:
    def test_require_osmnx_raises_when_missing(self):
        """_require_osmnx must raise ImportError with install instructions."""
        with patch.dict(sys.modules, {"osmnx": None}):  # type: ignore[dict-item]
            with pytest.raises(ImportError, match="pip install"):
                OSMImporter._require_osmnx()

    def test_from_coords_raises_when_osmnx_missing(self):
        """from_coords must surface a clear ImportError when osmnx is absent."""
        importer = OSMImporter()
        with patch.object(
            OSMImporter,
            "_require_osmnx",
            side_effect=ImportError("Install osmnx"),
        ):
            with pytest.raises(ImportError):
                importer.from_coords(48.86, 2.33, 48.85, 2.29)
