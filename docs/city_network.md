# City Road Network Descriptor

The file `tools/config/city_network.json` defines the hand-crafted road
network used as the default environment for the ARCO A* pipeline and the
Pygame real-time frontend.

## File format

The format is intentionally minimal — only fields needed to define graph
topology and edge geometry are included.

```json
{
  "nodes": [
    {"id": 0, "x": 365, "y": 110},
    ...
  ],
  "edges": [
    {"from": 0, "to": 1, "waypoints": [[395, 118], [425, 102]]},
    {"from": 0, "to": 5},
    ...
  ]
}
```

### Node object

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | integer | **yes** | Unique node identifier |
| `x` | number | **yes** | X-coordinate in meters |
| `y` | number | **yes** | Y-coordinate in meters |

### Edge object

| Field | Type | Required | Description |
|---|---|---|---|
| `from` | integer | **yes** | First endpoint node `id` |
| `to` | integer | **yes** | Second endpoint node `id` |
| `waypoints` | array of `[x, y]` | no | Ordered intermediate control points. Omit or use `[]` for a straight segment. |

Edges are **undirected**: a single entry covers both traversal directions.

## Waypoint convention

Waypoints are always stored in the **canonical direction** — going from the
lower node `id` to the higher node `id`. The loader enforces this automatically:
if a JSON edge defines `from > to`, the waypoints are reversed on load before
being passed to the graph.

`RoadGraph.full_edge_geometry(a, b)` returns waypoints in the correct order
for both `a < b` (forward) and `a > b` (reversed) traversal, ensuring smooth
S-curve paths in both directions with no back-and-forth motion.

### S-curve formula

All waypoints in `city_network.json` are generated with:

```
wp1 = A + (1/3)*(B-A) + d * perp(B-A)
wp2 = A + (2/3)*(B-A) - d * perp(B-A)
```

where `d = 0.09 * |AB|`, `perp(v)` is the left-perpendicular unit vector, and
`A`, `B` are the positions of the lower-id and higher-id nodes respectively.

Properties:
- **No back-and-forth**: both waypoints have projection `t ∈ (0, 1)` along the
  chord AB, so the path always progresses from A toward B.
- **Symmetric reversal**: traversing B→A with reversed waypoints yields a
  valid S-curve in the opposite direction.

## Network topology

The network has **61 nodes** and **110 edges** arranged as a triangular
(hexagonal-cell) mesh with four symmetric holes and four terminal nodes.

```
         [57]                ← terminal N (365, 1070)
          |
     ─ ─ ─ ─ ─ ─ ─ ─
    |  ┌───╲   ╱───┐  |
    | /  hole  hole  \ |
[60]─── mesh  mesh ───[58]  ← terminals W / E
    | \  hole  hole  / |
    |  └───╱   ╲───┘  |
     ─ ─ ─ ─ ─ ─ ─ ─
          |
         [59]                ← terminal S (365, -70)
```

### Node groups

| IDs | Description |
|---|---|
| 0–56 | **Mesh nodes** — hexagonal triangular lattice, radius ~415 m from center |
| 57 | **Terminal N** — (365, 1070) |
| 58 | **Terminal E** — (1085, 422) |
| 59 | **Terminal S** — (365, -70) |
| 60 | **Terminal W** — (−85, 422) |

### Mesh structure

- **Cell spacing**: 90 m horizontal, ≈78 m vertical (equilateral triangles)
- **Boundary**: elliptical, semi-axes ≈415 m
- **Holes**: 4 symmetric open regions (NE, SE, SW, NW quadrants), each
  removing ~6 nodes and ~10 edges from the lattice
- **Node degree distribution**: mostly 3–5 connections (interior nodes up to 6)

## Route variety

The two-ring-plus-mesh topology gives many geometrically distinct routes. For
example, from terminal E (58) to terminal W (60) A* must navigate across the
full width of the mesh, choosing between paths that go above or below the
central holes.

## Loading the network

```python
from arco.mapping.graph.loader import load_road_graph

graph = load_road_graph("tools/config/city_network.json")
print(len(graph.nodes), "nodes")   # 61
print(len(graph.edges), "edges")   # 110
```

The loader is implemented in `src/arco/mapping/graph/loader.py`.
