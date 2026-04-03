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
    {"id": 0, "x": 300, "y": 410},
    ...
  ],
  "edges": [
    {"from": 0, "to": 1, "waypoints": [[329, 406], [349, 382]]},
    {"from": 0, "to": 8},
    ...
  ]
}
```

### Node object

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | integer | **yes** | Unique node identifier |
| `x` | number | **yes** | X-coordinate in metres |
| `y` | number | **yes** | Y-coordinate in metres |

### Edge object

| Field | Type | Required | Description |
|---|---|---|---|
| `from` | integer | **yes** | Source node `id` |
| `to` | integer | **yes** | Destination node `id` |
| `waypoints` | array of `[x, y]` | no | Ordered intermediate control points between the two endpoint nodes. Omit or use `[]` for a straight segment. |

Edges are **undirected**: a single entry covers both traversal directions.

## Waypoint geometry

Each `[x, y]` waypoint is an intermediate control point along a road segment.
The rendering and planning layers draw/interpolate a polyline through:

```
[start_node] → waypoint_1 → waypoint_2 → ... → [end_node]
```

### S-curve convention

All waypoints in `city_network.json` are generated with the S-curve formula:

```
wp1 = A + (1/3)*(B-A) + d * perp(B-A)
wp2 = A + (2/3)*(B-A) - d * perp(B-A)
```

where `d = 0.08 * |AB|` and `perp(v)` is the left-perpendicular unit vector.
This ensures:

- **No back-and-forth**: both waypoints have projection parameter `t ∈ (0, 1)`
  onto the chord `AB`, so the path always progresses from A toward B.
- **Gentle S-curve**: the first third curves left, the second third curves right,
  creating a smooth, natural-looking shape without sharp turns.

## Network topology

The network has **20 nodes** and **40 edges** arranged in two concentric rings
with four terminal entry/exit nodes:

```
              [16]              ← terminal N (300, 590)
             /    \
           [8]    [9]           ← outer NNW / NNE
          / |      | \
        [7][0]--[0][1][2]       ← inner ring (r=110)
        ...  inner ring  ...
        [6]----[6]---[2]
          \ |      | /
          [15]   [10]           ← outer WNW / ENE
           |      |
          [14]  [11]            ← outer WSW / ESE
             \  /
             [13][12]           ← outer SSW / SSE
                |
               [18]             ← terminal S (300, 10)
```

*(Simplified — see JSON for exact positions.)*

### Node groups

| IDs | Description |
|---|---|
| 0–7 | **Inner ring** — 8 nodes at radius 110 m, centred at (300, 300) |
| 8–15 | **Outer ring** — 8 nodes at radius 220 m, angularly offset by 22.5° from inner ring |
| 16–19 | **Terminals** — 4 entry/exit nodes (N, E, S, W) at radius ~290 m |

### Edge groups

| Description | Count |
|---|---|
| Inner ring edges (closed ring) | 8 |
| Inner → outer radial connections (each inner node → 2 outer nodes) | 16 |
| Outer ring edges (closed ring) | 8 |
| Terminal → outer connections (each terminal → 2 outer nodes) | 8 |
| **Total** | **40** |

## Route variety

The two-ring topology gives multiple geometrically distinct routes between
any pair of terminals. For example, from terminal N (16) to terminal S (18):

| Route | Path | Character |
|---|---|---|
| Outer-ring East | 16→9→10→11→12→18 | Fast perimeter (5 nodes) |
| Outer-ring West | 16→8→15→14→13→18 | Symmetric perimeter (5 nodes) |
| Inner shortcut E | 16→9→(inner ring E side)→12→18 | Shorter via inner ring |
| Inner shortcut W | 16→8→(inner ring W side)→13→18 | Symmetric inner shortcut |
| Mixed | 16→8→0→1→…→4→12→18 | Cross-ring route |

A* with a Euclidean heuristic finds the optimal of these automatically.

## Loading the network

```python
from arco.mapping.graph.loader import load_road_graph

graph = load_road_graph("tools/config/city_network.json")
print(len(graph.nodes), "nodes")   # 20
print(len(graph.edges), "edges")   # 40
```

The loader is implemented in `src/arco/mapping/graph/loader.py`.
