"""Road-graph loading, re-exported from the compiled extension.

``load_road_graph`` parses the JSON network descriptor documented in
``docs/city_network.md`` and returns a populated
:class:`~arco.mapping.graph.road.RoadGraph`. The parser is compiled, so
reading a city network runs no Python.
"""

from arco._arco import load_road_graph

__all__ = ["load_road_graph"]
