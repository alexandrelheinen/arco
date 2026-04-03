"""RoadGraphLoader: load a RoadGraph from a JSON network descriptor file."""

from __future__ import annotations

import json
import os
from typing import Union

from .road import RoadGraph


def load_road_graph(path: Union[str, os.PathLike]) -> RoadGraph:
    """Load a :class:`RoadGraph` from a JSON network descriptor file.

    The descriptor format is documented in ``docs/city_network.md``.
    Each node must have ``id``, ``x``, and ``y`` fields.  Each edge must have
    ``from``, ``to``, and an optional ``waypoints`` list of ``[x, y]`` pairs.

    Args:
        path: Absolute or relative path to the ``.json`` descriptor file.

    Returns:
        A fully populated :class:`RoadGraph` ready for route planning.

    Raises:
        FileNotFoundError: When *path* does not point to an existing file.
        KeyError: When a required field is missing from a node or edge entry.
        ValueError: When a referenced node ID has not been added to the graph.

    Example::

        from arco.mapping.graph.loader import load_road_graph

        graph = load_road_graph("tools/config/city_network.json")
        print(len(graph.nodes), "nodes", len(graph.edges), "edges")
    """
    path = os.fspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Network descriptor not found: {path!r}")

    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    graph = RoadGraph()

    for node in data["nodes"]:
        graph.add_node(int(node["id"]), float(node["x"]), float(node["y"]))

    node_ids = set(graph.nodes)
    for edge in data["edges"]:
        src = int(edge["from"])
        dst = int(edge["to"])
        if src not in node_ids:
            raise ValueError(f"Edge references unknown node id {src!r}")
        if dst not in node_ids:
            raise ValueError(f"Edge references unknown node id {dst!r}")
        raw_wps = edge.get("waypoints", [])
        waypoints = [(float(wp[0]), float(wp[1])) for wp in raw_wps]
        # RoadGraph stores waypoints at canonical key (min_id, max_id).
        # full_edge_geometry() returns them in the min→max direction for
        # forward traversal and reverses them for reverse traversal.
        # So waypoints must always describe the min_id→max_id direction;
        # if the JSON defines from > to, reverse them here.
        if src > dst:
            waypoints = waypoints[::-1]
        graph.add_edge(src, dst, waypoints=waypoints)

    return graph
