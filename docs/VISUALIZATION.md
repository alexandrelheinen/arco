# Visualization Tools

ARCO ships one visualization tool: **arcosim** — the unified CLI for both
static image generation and real-time interactive simulation, driven by YAML
scenario files.

## arcosim — Unified Scenario Runner

`arcosim` runs a scenario through the ARCO pipeline. It supports two modes:

- **Real-time simulation** (default): pygame window with live animation.
- **Static image mode** (`--image` / `--static`): matplotlib figure, optionally
  saved to a file.

### Dependencies

```bash
# Static image mode only
pip install arco[tools]          # matplotlib + pyyaml; no pygame needed

# Real-time simulation
pip install arco[tools,pygame]   # adds pygame >= 2.0 and PyOpenGL >= 3.1
```

A display server (or virtual framebuffer such as `xvfb`) is required for
interactive use. For headless recording, `xvfb-run` and `ffmpeg` are needed.

### Usage

```bash
# Interactive simulation (requires pygame)
arcosim map/city.yml

# Record to MP4 (requires pygame + ffmpeg)
arcosim map/city.yml --record output/city.mp4

# Limit recording duration
arcosim map/city.yml --record output/city.mp4 --record-duration 30

# Fast headless recording: skip animated planner-tree reveal
arcosim map/city.yml -o output/city.mp4 -d 60 --fast-record

# Static image mode — opens matplotlib window
arcosim map/city.yml --image

# Static image mode — save to file (headless-safe)
arcosim map/city.yml --image --record output/city.png

# --static is an alias for --image
arcosim map/city.yml --static --record output/city.png
```

### Supported scenarios

| Scenario | Description |
|----------|-------------|
| `city`   | Neighborhood race — RRT* / SST / A* with shared NMPCC |
| `ppp`    | PPP gantry warehouse — joint-space MPC |
| `rrp`    | RRP SCARA arm — joint-space MPC |
| `occ`    | Piano-movers — multi-actuator object transport |

### Shared presentation chrome

All four scenarios compose the same shell via
`arco.simulator.sim.layout.ScreenLayout`:

- **Header** — left-aligned phase title (`City · race`, `PPP · path reveal`, …)
  plus a thin **method accent stripe** (RRT* / SST / A* colors from the palette)
- **Sidebar** — compact planner summary or race standings with method accent bars
- **Footer** — controls / phase hint
- **Content** — scenario viewport (city follow-cam, PPP/RRP 3-D, OCC dual panels)

Chrome colors live under `ui.chrome_*` in `src/arco/config/colors.yml`.

### Release / CI video generation

`scripts/generate_videos.sh --release` (used by `.github/workflows/release.yml`):

- Remaps logical scenario `city` → `map/city_mpc_preview.yml` (reduced RRT*/SST/A*
  sample budgets) while still writing `arcosim_city.mp4`.
- Passes `--fast-record` so recordings skip tree-reveal pacing and spend the
  duration budget on the race / tracking phase.
- Uses **45 s** clips: at 30 fps × 0.1 s sim steps that is 135 simulated
  seconds — enough for every racer (A* needs ~115 s on its windy grid
  route) to reach the goal on camera.
- Caches pip and installs CasADi (`arco[mpc]`) only for scenarios whose YAML
  sets `tracker: mpc` (`city`, `ppp`, `rrp`).

City race notes when `simulator.tracker: mpc`:

- Scenario YAML may set `simulator.mpc.horizon.{step_count,dt}` (city default
  is **50 × 0.1 s = 5.0 s**; the model dt **must** equal the 0.1 s simulator
  timestep — see [control_mpcc.md](control_mpcc.md)).  The tracker is a
  classical **MPCC**: virtual progress speed capped by the curve-limited
  cruise, structural lag coupling (`lag: 6`), linear progress reward
  (`progress: 8`), strict quadratic contour (`contour: 10`,
  `contour_deadzone: 0` — flat bands invite equal-cost chatter), light
  heading alignment (`heading: 1`), soft obstacle barriers
  (`obstacle: 120`).  Full post-plan parameter inventory:
  [control_tracking_params.md](control_tracking_params.md); closed-loop
  quality gate: `tools/city_tracking_report.py`.
- The race renderer draws each racer's **MPC predicted XY polyline** (no tip
  disc) instead of a Pure-Pursuit carrot.
- Racers are **GTA2-style top-down car sprites**
  (`arco.simulator.sim.car_sprite`, `8.0 × 3.6` m half-extents): pixel-art
  body in each planner's color with dark glass, tires, head/taillights,
  drawn as oriented textured quads over the warm SDF road field.  Dim
  planned underlay (`2.0` px @ 0.35 α), bold executed past trails
  (`4.5` px), and prediction polylines (`3.5` px) complete the look.
- **Presentation (race phase):** follow-cam zooms to the pack (~200 m window,
  smoothed chase), a corner **minimap** keeps the full 600 m city, the sidebar
  switches to **standings** (place / gap / speed), and the header title flips
  from planning reveal → race · follow cam.  Planning reveal still uses the
  full bird's-eye map.

---

## Scenario YAML format

Each scenario YAML file begins with a `scenario:` key that names the scenario,
followed by scenario-specific parameters:

```yaml
scenario: city

# Example: city scenario parameters
start_node: 59
goal_node: 57
```

The YAML files for the built-in scenarios live in `map/`.

---

## Running smoke validation locally

```bash
# Short headless recording (requires xvfb + ffmpeg)
bash scripts/run_smoke_test.sh city
```
