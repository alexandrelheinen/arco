# ARCO — Pygame Real-Time Simulation

Real-time animation of the planning + tracking pipeline for three planners:
**A\***, **RRT\***, and **SST**.  Each script follows the same two-phase
structure: reveal the planning result, then run the Dubins vehicle.

## What is shown

### Phase 1 — Background reveal

| Planner | Background layer |
|---|---|
| A\* | Static road network with planned route (no animation) |
| RRT\* | Exploration tree growing node-by-node, then final path |
| SST | Exploration tree (teal) growing node-by-node, then final path |

### Phase 2 — Vehicle tracking (all planners)

| Layer | Colour |
|---|---|
| Planned path | Orange (A\*) / Yellow (RRT\*/SST) |
| Past vehicle trajectory | Blue |
| Lookahead tracking target | Yellow circle |
| Vehicle (rectangle + heading arrow) | Green |

## Requirements

```bash
pip install -r requirements.txt
```

This installs **pygame ≥ 2.0** and **numpy**.  The ARCO library must also be
installed or importable (see the repository root README).

```bash
# From the repository root, install everything:
pip install -e ".[tools]"
```

## Running

```bash
# From the repository root:
python tools/simulator/main/astar.py
python tools/simulator/main/rrt.py
python tools/simulator/main/sst.py
```

### Optional flags (identical across all entrypoints)

| Flag | Default | Description |
|---|---|---|
| `--fps N` | `30` | Target animation frame rate |
| `--dt S` | `0.1` | Simulation timestep per frame (seconds) |
| `--camera {full,follow}` | `full` | Starting camera: whole-scene view or vehicle-following |
| `--zoom` | off | Fit initial view to planned path bounding box |
| `--record FILE` | _(none)_ | Record headless MP4 to FILE and exit (requires `ffmpeg`) |
| `--record-duration S` | `60` | Maximum clip length in seconds |

### Recording examples

```bash
# A* full-network view:
python tools/simulator/main/astar.py --record astar.mp4

# RRT* zoomed to path bounding box:
python tools/simulator/main/rrt.py --zoom --record rrt_zoom.mp4

# SST follow-vehicle camera, 30-second clip at 60 fps:
python tools/simulator/main/sst.py --camera follow --fps 60 \
    --record-duration 30 --record sst_follow.mp4
```

## Keyboard controls

| Key | Action |
|---|---|
| `SPACE` | Pause / resume simulation |
| `R` | Restart current phase |
| `C` | Toggle camera mode (full / follow vehicle) |
| `+` / `-` | Zoom in / out (follow-vehicle mode only) |
| `Q` / `Escape` | Quit |

## Architecture

```
tools/simulator/
├── main/
│   ├── astar.py        — thin entrypoint (~35 lines): parse CLI, build AStarScene, call run_sim
│   ├── rrt.py          — thin entrypoint: build RRTScene, call run_sim
│   └── sst.py          — thin entrypoint: build SSTScene, call run_sim
├── env/
│   ├── astar.py        — AStarScene: road network + A* planning
│   ├── rrt.py          — RRTScene: obstacle field + RRT* planning, tree reveal
│   └── sst.py          — SSTScene: obstacle field + SST planning, tree reveal
├── sim/
│   ├── loop.py         — run_sim(): unified two-phase game loop
│   ├── scene.py        — SimScene ABC
│   ├── tracking.py     — VehicleConfig, build_vehicle_sim(), find_lookahead()
│   ├── camera.py       — FollowTransform, CameraFilter (2nd-order spring)
│   └── video.py        — VideoWriter (ffmpeg MP4 recorder)
├── renderer.py         — stateless drawing primitives (road, tree, path, HUD)
├── requirements.txt
└── README.md
```

### Two-phase loop

```
run_sim(scene)
  ├── scene.build()               ← planner runs once, after pygame.init()
  ├── Phase 1 (background):
  │     scene.draw_background(surface, transform, revealed)   ← incremental
  │     scene.draw_background_hud(surface, font, revealed)
  │     (background_total == 0 → skip to Phase 2 immediately)
  └── Phase 2 (tracking):
        scene.draw_background(surface, transform, total)      ← static
        draw_trajectory / draw_tracking_target / draw_vehicle
        draw_tracking_hud
```

**Backend independence**: all planning and control logic lives in `arco.*`.
`renderer.py` is a *stateless* drawing adapter.  `sim/loop.py` contains no
planner-specific code.  Adding a new planner requires only a new `env/*.py`
file that implements `SimScene`.

## Configuration

| File | Controls |
|---|---|
| `tools/config/graph.yml` | Road network layout (A\*) |
| `tools/config/rrt.yml` | RRT\* bounds, sample count, obstacle field |
| `tools/config/sst.yml` | SST bounds, sample count, witness radius |
| `tools/config/vehicle.yml` | Dubins vehicle parameters (A\*) |
