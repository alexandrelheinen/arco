# ARCO — Pygame Real-Time Simulation

Real-time animation of the **A* horse auto-follow pipeline** on the
hand-crafted city road network, rendered with [Pygame](https://www.pygame.org/).

## What is shown

| Layer | Colour |
|---|---|
| Road network edges (with waypoint curves) | Gray |
| Planned A* route edges | Red/orange |
| Smooth interpolated path | Dashed orange |
| Past vehicle trajectory | Blue |
| Lookahead tracking target | Yellow circle |
| Vehicle (rectangle + heading arrow) | Green |

## Requirements

```bash
pip install -r requirements.txt
```

This installs **pygame ≥ 2.0** only.  The ARCO library itself must also be
installed or importable (see the repository root README).

```bash
# From the repository root, install everything at once:
pip install -e ".[tools]"
pip install pygame
```

## Running

```bash
# From the repository root:
python tools/simulator/main.py

# Or from this folder:
cd tools/simulator
python main.py
```

### Optional flags

| Flag | Default | Description |
|---|---|---|
| `--fps N` | `30` | Cap the animation frame rate |
| `--dt S` | `0.1` | Simulation timestep per frame (seconds) |

## Keyboard controls

| Key | Action |
|---|---|
| `SPACE` | Pause / resume simulation |
| `R` | Restart from the beginning |
| `Q` / `Escape` | Quit |

## Architecture

```
tools/simulator/
├── main.py        — entry point: event loop, simulation stepping, render calls
├── renderer.py    — pure drawing functions (no simulation logic)
├── requirements.txt
└── README.md
```

`main.py` orchestrates the simulation:

1. Generates a road graph from `tools/config/graph.yml` via `graph.generator`.
2. Plans an A* route between farthest graph nodes using `arco.planning.discrete.RouteRouter`.
3. Builds a dense smooth path from edge waypoints.
4. Drives a `arco.guidance.vehicle.DubinsVehicle` with a
   `arco.guidance.pure_pursuit.PurePursuitController` at each frame.
5. Calls `renderer.*` to draw the current state onto the Pygame surface.

**Backend independence**: all planning and control logic lives in `arco.*`.
`renderer.py` is a *stateless* drawing adapter; it only receives data from
`main.py` and calls Pygame primitives.  Swapping the GUI framework requires
touching only `renderer.py` and `main.py`.

## Network configuration

The road network generation is configured in `tools/config/graph.yml`.
