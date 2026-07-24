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
- Uses **30 s** clips (not 60 s): with fast-record every frame is
  race/tracking; 30 s matches the race portion of the former 60 s videos
  that spent ~half the time revealing trees.
- Caches pip and installs CasADi (`arco[mpc]`) only for scenarios whose YAML
  sets `tracker: mpc` (`city`, `ppp`, `rrp`).

City race notes when `simulator.tracker: mpc`:

- Scenario YAML may set `simulator.mpc.horizon.{step_count,dt}` (city default
  is **72 × 0.05 s = 3.6 s**, ~43 m at the soft 12 m/s cruise ≈ half a block).
  City uses **progress-first** contouring with **`contour_deadzone: 0`**
  (`lag: 4`, `control: 0.5`, `obstacle: 120`) plus soft but lane-viable
  turn limits (ω̇ 90 °/s²): planners ignore dynamics, so lag/progress trade
  against contour to widen infeasible kinks **inside the road corridor**
  while `s` advances (see `tools/mpc_progress_first_demo.py`).  A non-zero
  deadzone flat-zones the lateral cost (zero gradient → equal-cost left/right
  chatter / wobble); a prior 8 m band also ate planner clearance.  An
  over-stiff retune (`control`/`obstacle` too high, ω̇ too low) understeered
  then hunted — avoid that.  Full post-plan parameter inventory:
  [control_tracking_params.md](control_tracking_params.md).
- The race renderer draws each racer's **MPC predicted XY polyline** (no tip
  disc) instead of a Pure-Pursuit carrot.
- Race glyphs stay **oriented rectangles** (`8.0 × 3.6` m half-extents) over
  the warm SDF road field: dim planned underlay (`2.0` px @ 0.35 α), bold
  executed past trails (`4.5` px), and prediction polylines (`3.5` px).
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
