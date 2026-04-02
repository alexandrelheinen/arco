"""OSMImporter: downloads and parses OpenStreetMap networks into RoadGraph."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from ..graph.road import RoadGraph

logger = logging.getLogger(__name__)

_LARGE_GRAPH_THRESHOLD = 2_000
# Approximate metres per degree of latitude/longitude (safe for ≤ 5 km margins)
_METRES_PER_DEGREE = 111_000.0


class OSMImporter:
    """Downloads and parses a real-world OSM street network into a RoadGraph.

    Wraps the ``osmnx`` library (optional dependency; install with
    ``pip install arco[osm]``) to fetch, project, and convert an
    OpenStreetMap graph for use with
    :class:`~arco.planning.discrete.RouteRouter`.

    Both entry points apply the same internal pipeline:

    1. Resolve start/goal to WGS-84 (lat, lon).
    2. Compute a padded bounding box.
    3. Download the OSM graph via the Overpass API.
    4. Extract the largest weakly-connected component.
    5. Reproject to a local UTM zone (all distances in metres).
    6. Shift the coordinate origin to ``(0, 0)`` for numerical stability.
    7. Remap OSM node IDs to 0-based contiguous integers.
    8. Build and return a :class:`~arco.mapping.graph.road.RoadGraph`.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def from_addresses(
        self,
        address_start: str,
        address_goal: str,
        margin_m: float = 500.0,
        network_type: str = "bike",
    ) -> Tuple[RoadGraph, Tuple[float, float], Tuple[float, float]]:
        """Download and parse the OSM network between two addresses.

        Geocodes both addresses via Nominatim (bundled in osmnx), computes
        the bounding rectangle in WGS-84, pads by *margin_m*, downloads
        the filtered street graph, projects to UTM (metres), prunes to the
        largest connected component, and converts to a
        :class:`~arco.mapping.graph.road.RoadGraph`.

        Args:
            address_start: Human-readable address string, e.g.
                ``"Musée du Louvre, Paris, France"``.
            address_goal: Human-readable address string, e.g.
                ``"Tour Eiffel, Paris, France"``.
            margin_m: Padding around the bounding rectangle in metres.
            network_type: osmnx network filter: ``"bike"``, ``"walk"``,
                or ``"drive"``.

        Returns:
            Tuple of ``(graph, start_xy, goal_xy)`` where *start_xy* and
            *goal_xy* are projected metric coordinates within the returned
            graph.

        Raises:
            ImportError: If ``osmnx`` is not installed.
            ValueError: If either address cannot be geocoded by Nominatim.
        """
        ox = self._require_osmnx()

        try:
            lat_s, lon_s = ox.geocode(address_start)
        except Exception as exc:
            raise ValueError(
                f"Failed to geocode '{address_start}': {exc}"
            ) from exc

        try:
            lat_g, lon_g = ox.geocode(address_goal)
        except Exception as exc:
            raise ValueError(
                f"Failed to geocode '{address_goal}': {exc}"
            ) from exc

        logger.info(
            "Geocoded start: (%.6f, %.6f)  goal: (%.6f, %.6f)",
            lat_s,
            lon_s,
            lat_g,
            lon_g,
        )

        return self.from_coords(
            lat_s, lon_s, lat_g, lon_g, margin_m, network_type
        )

    def from_coords(
        self,
        lat_start: float,
        lon_start: float,
        lat_goal: float,
        lon_goal: float,
        margin_m: float = 500.0,
        network_type: str = "bike",
    ) -> Tuple[RoadGraph, Tuple[float, float], Tuple[float, float]]:
        """Download and parse the OSM network between two GPS coordinates.

        Args:
            lat_start: Latitude of the start point (decimal degrees).
            lon_start: Longitude of the start point (decimal degrees).
            lat_goal: Latitude of the goal point (decimal degrees).
            lon_goal: Longitude of the goal point (decimal degrees).
            margin_m: Padding around the bounding rectangle in metres.
            network_type: osmnx network filter: ``"bike"``, ``"walk"``,
                or ``"drive"``.

        Returns:
            Tuple of ``(graph, start_xy, goal_xy)`` in projected metric
            coordinates.

        Raises:
            ImportError: If ``osmnx`` is not installed.
        """
        ox = self._require_osmnx()

        # 1. Compute padded bounding box ----------------------------------
        delta = margin_m / _METRES_PER_DEGREE
        min_lat = min(lat_start, lat_goal) - delta
        max_lat = max(lat_start, lat_goal) + delta
        min_lon = min(lon_start, lon_goal) - delta
        max_lon = max(lon_start, lon_goal) + delta

        logger.info(
            "Bounding box: lat=[%.6f, %.6f]  lon=[%.6f, %.6f]",
            min_lat,
            max_lat,
            min_lon,
            max_lon,
        )

        # 2. Download OSM graph -------------------------------------------
        G_raw = ox.graph_from_bbox(
            north=max_lat,
            south=min_lat,
            east=max_lon,
            west=min_lon,
            network_type=network_type,
            simplify=True,
            retain_all=False,
        )
        logger.info(
            "Downloaded: %d nodes, %d edges",
            G_raw.number_of_nodes(),
            G_raw.number_of_edges(),
        )

        # 3. Largest weakly-connected component ---------------------------
        G_raw = self._largest_component(ox, G_raw)
        logger.info("After LCC pruning: %d nodes", G_raw.number_of_nodes())

        # 4. Project to UTM -----------------------------------------------
        G_proj = ox.project_graph(G_raw)

        # 5. Compute origin shift -----------------------------------------
        xs = [data["x"] for _, data in G_proj.nodes(data=True)]
        ys = [data["y"] for _, data in G_proj.nodes(data=True)]
        origin_x = min(xs)
        origin_y = min(ys)

        logger.info(
            "UTM range: x=[%.1f, %.1f]  y=[%.1f, %.1f]",
            min(xs),
            max(xs),
            min(ys),
            max(ys),
        )

        # 6. Build RoadGraph with remapped node IDs -----------------------
        node_ids = list(G_proj.nodes())
        id_map: Dict[int, int] = {osmid: i for i, osmid in enumerate(node_ids)}

        road_graph = RoadGraph()

        for osmid, data in G_proj.nodes(data=True):
            road_graph.add_node(
                id_map[osmid],
                data["x"] - origin_x,
                data["y"] - origin_y,
            )

        for u, v, data in G_proj.edges(data=True):
            if u not in id_map or v not in id_map:
                continue
            length = float(data.get("length", 1.0))
            geometry = data.get("geometry", None)
            waypoints: List[Tuple[float, float]] = []
            if geometry is not None:
                coords = list(geometry.coords)
                waypoints = [
                    (c[0] - origin_x, c[1] - origin_y) for c in coords[1:-1]
                ]
            road_graph.add_edge(
                id_map[u], id_map[v], weight=length, waypoints=waypoints
            )

        n_nodes = len(road_graph.nodes)
        n_edges = len(road_graph.edges)
        logger.info("RoadGraph: %d nodes, %d edges", n_nodes, n_edges)

        if n_nodes > _LARGE_GRAPH_THRESHOLD:
            logger.warning(
                "Large graph (%d nodes). A* may be slow. "
                "Consider reducing margin_m.",
                n_nodes,
            )

        # 7. Project start / goal into the metric coordinate system ------
        crs: str = G_proj.graph["crs"]
        start_xy = self._project_latlon(
            lat_start, lon_start, crs, origin_x, origin_y
        )
        goal_xy = self._project_latlon(
            lat_goal, lon_goal, crs, origin_x, origin_y
        )

        logger.info(
            "Start: (%.1f, %.1f)  Goal: (%.1f, %.1f)",
            start_xy[0],
            start_xy[1],
            goal_xy[0],
            goal_xy[1],
        )

        return road_graph, start_xy, goal_xy

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _require_osmnx():
        """Return the osmnx module, or raise ImportError if unavailable.

        Returns:
            The ``osmnx`` module object.

        Raises:
            ImportError: If ``osmnx`` is not installed.
        """
        try:
            import osmnx as ox  # noqa: PLC0415

            return ox
        except ImportError as exc:
            raise ImportError(
                "osmnx is required for OSMImporter. "
                "Install it with: pip install 'arco[osm]'"
            ) from exc

    @staticmethod
    def _largest_component(ox_module, G):
        """Return the largest weakly-connected component of *G*.

        Tries the osmnx 1.x built-in utility first, then falls back to a
        direct networkx call for other API versions.

        Args:
            ox_module: The osmnx module.
            G: A networkx MultiDiGraph returned by osmnx.

        Returns:
            A subgraph containing only the largest weakly-connected
            component.
        """
        try:
            return ox_module.utils_graph.get_largest_component(
                G, strongly=False
            )
        except AttributeError:
            import networkx as nx  # noqa: PLC0415

            nodes = max(nx.weakly_connected_components(G), key=len)
            return G.subgraph(nodes).copy()

    @staticmethod
    def _project_latlon(
        lat: float,
        lon: float,
        crs: str,
        origin_x: float,
        origin_y: float,
    ) -> Tuple[float, float]:
        """Project a WGS-84 coordinate into the graph metric CRS.

        Args:
            lat: Latitude in decimal degrees.
            lon: Longitude in decimal degrees.
            crs: Target coordinate reference system string, e.g.
                ``"EPSG:32631"``.
            origin_x: X-axis origin offset to subtract (metres).
            origin_y: Y-axis origin offset to subtract (metres).

        Returns:
            ``(x, y)`` position in the shifted metric coordinate system.
        """
        from pyproj import Transformer  # noqa: PLC0415

        transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        utm_x, utm_y = transformer.transform(lon, lat)
        return (utm_x - origin_x, utm_y - origin_y)

    @staticmethod
    def _compute_bbox(
        lat_start: float,
        lon_start: float,
        lat_goal: float,
        lon_goal: float,
        margin_m: float,
    ) -> Tuple[float, float, float, float]:
        """Compute a padded WGS-84 bounding box.

        Args:
            lat_start: Latitude of the start point (decimal degrees).
            lon_start: Longitude of the start point (decimal degrees).
            lat_goal: Latitude of the goal point (decimal degrees).
            lon_goal: Longitude of the goal point (decimal degrees).
            margin_m: Padding in metres to add on all sides.

        Returns:
            ``(min_lat, max_lat, min_lon, max_lon)`` in decimal degrees.
        """
        delta = margin_m / _METRES_PER_DEGREE
        min_lat = min(lat_start, lat_goal) - delta
        max_lat = max(lat_start, lat_goal) + delta
        min_lon = min(lon_start, lon_goal) - delta
        max_lon = max(lon_start, lon_goal) + delta
        return min_lat, max_lat, min_lon, max_lon
