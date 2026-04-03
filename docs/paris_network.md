# Paris Downtown Road Network Descriptor

The file `tools/config/paris_network.json` defines the hand-crafted road
network used as the default environment for the ARCO A* pipeline and the
Pygame real-time frontend.

## Motivation

The procedural `RoadNetworkGenerator` was removed because it produced
networks that were difficult to control visually and did not yield the
organic, city-like layout required for the horse auto-follow scenario.
The hand-crafted descriptor gives full control over topology, geometry,
and labelling, while remaining reusable across multiple scripts.

## File format

```jsonc
{
  "name":             "<human-readable name>",
  "description":      "<free-text description>",
  "coordinate_unit":  "metres",           // informational only

  "nodes": [
    { "id": 0, "x": 350.0, "y": 400.0, "label": "Île de la Cité Ouest" },
    ...
  ],

  "edges": [
    {
      "from": 0,
      "to":   4,
      "label": "Pont Saint-Michel",
      "waypoints": [ [328.0, 382.0], [307.0, 362.0] ]
    },
    ...
  ]
}
```

### Top-level fields

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | no | Short human-readable name |
| `description` | string | no | Longer description |
| `coordinate_unit` | string | no | Physical unit of `x`/`y` (informational) |
| `nodes` | array | **yes** | List of intersection node objects |
| `edges` | array | **yes** | List of road segment edge objects |

### Node object

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | integer | **yes** | Unique node identifier |
| `x` | number | **yes** | X-coordinate in `coordinate_unit` |
| `y` | number | **yes** | Y-coordinate in `coordinate_unit` |
| `label` | string | no | Human-readable street/place name |

### Edge object

| Field | Type | Required | Description |
|---|---|---|---|
| `from` | integer | **yes** | Source node `id` |
| `to` | integer | **yes** | Destination node `id` |
| `label` | string | no | Human-readable road name |
| `waypoints` | array of `[x, y]` | no | Ordered intermediate control points between the two nodes (not including the nodes themselves). Empty list means a straight segment. |

Edges are **undirected**: a single entry covers both traversal directions.
The `weight` of each edge defaults to the Euclidean distance between the two
endpoint node positions (plus the sum of segment lengths through the
waypoints is *not* currently used as the weight — only the straight-line
endpoint distance is, matching the behaviour of `RoadGraph.add_edge`).

## Waypoint geometry

Each waypoint `[x, y]` is an intermediate control point along the road
segment.  The rendering layer (`tools/viewer/road.py` and
`tools/pygame_sim/renderer.py`) draws a polyline through:

```
[start_node] → waypoint_1 → waypoint_2 → ... → [end_node]
```

Waypoints are stored in "source→destination" order (i.e., in the order that
traverses from the lower-`id` node to the higher-`id` node).  When a path is
traversed in the reverse direction `full_edge_geometry` automatically reverses
the waypoint list.

### Design guidelines for waypoints

- **No back-and-forth**: waypoints must lie strictly between the two endpoint
  nodes when projected onto the chord.  A waypoint that doubles back creates
  a loop, which the Pure Pursuit controller handles poorly.
- **Gentle curves**: offset each waypoint no more than ~15 % of the chord
  length perpendicular to the chord.
- **S-curves**: use two waypoints with opposite lateral offsets (first
  deviates left, second deviates right, or vice versa) to produce a natural
  S-curve without any backward component.

## Paris downtown layout

The network has 20 nodes and 35 edges and is loosely inspired by the centre
of Paris:

```
         [18] ── [17] ── [16]
          │                │
         [8]  [9]  [10] [11] [15]
          │    │    │    │    │
 [19]────[0]──[1]──[3]──[6]──[14]
          │    │    │
         [4]──[5]──[2]
          │    │    │
         [7] [13] [12]
               │
              [13]──[14]
```

*(Approximate — see the JSON for exact positions.)*

### Node groups

| IDs | Description |
|---|---|
| 0–3 | **Île de la Cité** — central island in the Seine |
| 4–7 | **Left Bank** (Rive Gauche) — south of the river |
| 8–11 | **Right Bank** (Rive Droite) — north of the river |
| 12–19 | **Outer ring** — peripheral boulevards and gates |

### Edge groups

| Description | Count |
|---|---|
| Island-internal streets | 4 |
| Bridges Left Bank → Island | 3 |
| Bridges Right Bank → Island | 3 |
| Left Bank internal roads | 8 |
| Outer Left Bank roads | 2 |
| Right Bank internal roads | 7 |
| Right Bank outer roads | 3 |
| Outer / west connections | 2 |
| Outer east connections | 3 |
| **Total** | **35** |

## Loading the network

```python
from arco.mapping.graph.loader import load_road_graph

graph = load_road_graph("tools/config/paris_network.json")
print(len(graph.nodes), "nodes")   # 20
print(len(graph.edges), "edges")   # 35
```

The loader is implemented in `src/arco/mapping/graph/loader.py`.
