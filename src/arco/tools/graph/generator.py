"""Procedural graph generator for ARCO tools.

Generates :class:`~arco.mapping.graph.RoadGraph` instances from a
configuration dictionary.  The ``type`` key selects the generation algorithm.

Currently supported types:

- ``ring`` — irregular triangular mesh with S-shaped edges and symmetric holes.

Usage as a script::

    python tools/graph/generator.py
    python tools/graph/generator.py --save output/ring.json
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

try:
    from scipy.spatial import Delaunay
except ImportError as exc:
    raise ImportError(
        "scipy is required for procedural graph generation. "
        "Install it with: pip install 'arco[tools]'"
    ) from exc

from arco.mapping.graph import RoadGraph  # noqa: E402


def generate_graph(cfg: dict[str, Any]) -> RoadGraph:
    """Generate a :class:`~arco.mapping.graph.RoadGraph` from a config dict.

    The ``type`` key in *cfg* selects the generation algorithm.  All other
    keys are forwarded to the selected generator as parameters.

    Args:
        cfg: Configuration dictionary.  Must contain ``type`` (str).
            See ``tools/config/graph.yml`` for the full parameter reference.

    Returns:
        A populated :class:`~arco.mapping.graph.RoadGraph`.

    Raises:
        ValueError: When ``type`` is not a recognized generator name.
        KeyError: When a required parameter is absent from *cfg*.
    """
    graph_type = cfg.get("type", "ring")
    if graph_type == "ring":
        return _generate_ring(cfg)
    raise ValueError(f"Unknown graph type: {graph_type!r}. Supported: 'ring'.")


# ---------------------------------------------------------------------------
# Ring generator
# ---------------------------------------------------------------------------


def _generate_ring(cfg: dict[str, Any]) -> RoadGraph:
    """Generate an irregular triangular mesh with S-shaped edges and holes.

    The mesh is built on a jittered hexagonal lattice, triangulated with
    Delaunay, and thinned by removing edges that span across holes or exceed
    1.7 × ``mean_edge_length``.  Each surviving edge receives two S-curve
    waypoints using the formula from ``docs/city_network.md``.

    Args:
        cfg: Must contain ``width``, ``height``, ``mean_edge_length``,
            ``hole_count``, ``hole_radius``, ``curvature``.
            Optional: ``seed`` (int or null).

    Returns:
        A populated :class:`~arco.mapping.graph.RoadGraph`.
    """
    width = float(cfg["width"])
    height = float(cfg["height"])
    mean_edge_length = float(cfg["mean_edge_length"])
    hole_count = int(cfg["hole_count"])
    hole_radius = float(cfg["hole_radius"])
    curvature = float(cfg["curvature"])
    seed = cfg.get("seed")

    rng = np.random.default_rng(seed)

    # ── Jittered hexagonal lattice ─────────────────────────────────────────
    row_spacing = mean_edge_length * math.sqrt(3) / 2
    col_spacing = mean_edge_length
    positions: list[tuple[float, float]] = []
    row = 0
    y = 0.0
    while y <= height:
        col_offset = (col_spacing / 2) if (row % 2 == 1) else 0.0
        x = col_offset
        while x <= width:
            jx = float(rng.normal(0.0, 0.15 * mean_edge_length))
            jy = float(rng.normal(0.0, 0.15 * mean_edge_length))
            px = float(np.clip(x + jx, 0.0, width))
            py = float(np.clip(y + jy, 0.0, height))
            positions.append((px, py))
            x += col_spacing
        y += row_spacing
        row += 1

    # ── Hole placement ─────────────────────────────────────────────────────
    cx, cy = width / 2.0, height / 2.0
    hole_centers: list[tuple[float, float]] = [(cx, cy)]
    if hole_count > 1:
        placement_radius = 0.3 * min(width, height)
        theta_0 = float(rng.uniform(0.0, 2 * math.pi))
        for k in range(hole_count - 1):
            angle = theta_0 + k * 2 * math.pi / (hole_count - 1)
            hole_centers.append(
                (
                    cx + placement_radius * math.cos(angle),
                    cy + placement_radius * math.sin(angle),
                )
            )

    # ── Remove nodes inside holes ──────────────────────────────────────────
    positions = [
        (px, py)
        for px, py in positions
        if not any(
            math.hypot(px - hx, py - hy) < hole_radius
            for hx, hy in hole_centers
        )
    ]

    # ── Delaunay triangulation ─────────────────────────────────────────────
    pts = np.array(positions)
    tri = Delaunay(pts)

    edge_set: set[tuple[int, int]] = set()
    for simplex in tri.simplices:
        a, b, c = int(simplex[0]), int(simplex[1]), int(simplex[2])
        for u, v in ((a, b), (b, c), (a, c)):
            edge_set.add((min(u, v), max(u, v)))

    # ── Prune long edges ───────────────────────────────────────────────────
    max_edge_length = 1.7 * mean_edge_length
    edges: list[tuple[int, int]] = [
        (u, v)
        for u, v in edge_set
        if math.hypot(
            positions[u][0] - positions[v][0],
            positions[u][1] - positions[v][1],
        )
        <= max_edge_length
    ]

    # ── Build RoadGraph with S-curve waypoints ─────────────────────────────
    # Waypoints stored in canonical direction (u < v, already guaranteed above).
    #
    # Formula (per docs/city_network.md):
    #   perp = left-perpendicular unit vector of AB
    #   s    = random sign, independent per edge
    #   wp1  = A + (1/3)(B-A) + s * curvature * |AB| * perp
    #   wp2  = A + (2/3)(B-A) - s * curvature * |AB| * perp
    graph = RoadGraph()
    for node_id, (px, py) in enumerate(positions):
        graph.add_node(node_id, px, py)

    for u, v in edges:
        ax, ay = positions[u]
        bx, by = positions[v]
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)
        perp_x = -dy / length
        perp_y = dx / length
        s = float(rng.choice(np.array([-1.0, 1.0])))
        disturbance = s * curvature * length
        graph.add_edge(
            u,
            v,
            waypoints=[
                (
                    ax + dx / 3.0 + disturbance * perp_x,
                    ay + dy / 3.0 + disturbance * perp_y,
                ),
                (
                    ax + 2.0 * dx / 3.0 - disturbance * perp_x,
                    ay + 2.0 * dy / 3.0 - disturbance * perp_y,
                ),
            ],
        )

    return graph


# ---------------------------------------------------------------------------
# Script entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    from config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save",
        metavar="PATH",
        default=None,
        help=(
            "Save the generated graph as a JSON file loadable by "
            "load_road_graph()."
        ),
    )
    args = parser.parse_args()

    cfg = load_config("astar").get("graph", {})
    graph = generate_graph(cfg)
    print(
        f"Generated '{cfg.get('type', 'ring')}' graph: "
        f"{len(graph.nodes)} nodes, {len(graph.edges)} edges"
    )

    if args.save:
        data = {
            "nodes": [
                {
                    "id": n,
                    "x": graph.position(n)[0],
                    "y": graph.position(n)[1],
                }
                for n in graph.nodes
            ],
            "edges": [
                {
                    "from": a,
                    "to": b,
                    "waypoints": [
                        list(wp) for wp in graph.edge_geometry(a, b)
                    ],
                }
                for a, b, _ in graph.edges
            ],
        }
        abs_path = os.path.abspath(args.save)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        print(f"Saved to {args.save}")
