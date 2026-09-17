"""Road graph, re-exported from the compiled extension.

The implementation is ``RoadGraph`` in the ``arco-mapping`` crate,
reaching Python through :mod:`arco._arco`. It owns a Cartesian graph plus
the per-edge geometry. Reading a road graph out of a JSON descriptor has
no crate-side counterpart and stays in :mod:`arco.mapping.graph.loader`.
"""

from arco._arco import RoadGraph

__all__ = ["RoadGraph"]
