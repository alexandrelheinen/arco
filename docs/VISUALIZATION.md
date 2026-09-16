# Visualization Tools

ARCO ships one visualization tool: **arcosim**, the CLI that runs a YAML
scenario through the ARCO pipeline and renders it.

## arcosim — Unified Scenario Runner

`arcosim` has one renderer, the pygame / OpenGL one, and three output modes:

- **Interactive** (default): a window with live animation.
- **Video** (`-o FILE.mp4`): headless MP4 recording through ffmpeg.
- **Still frames** (`-o FILE.png --still FRAME`): the same render, with the
  selected frames written as PNG instead of encoded. See
  [still_frames.md](still_frames.md).

The matplotlib layer under `src/arco/simulator/viewer/` is a library used by
the pipeline frontends; it has no `arcosim` entry point. Earlier revisions of
this file documented `arcosim --image` and `arcosim --static` for it. Those
flags were never implemented and have been removed from the documentation
rather than from the CLI, which never had them.

### Dependencies

```bash
pip install arco[tools,pygame]   # pyyaml + pygame >= 2.0 + PyOpenGL >= 3.1
pip install arco[mpc]            # CasADi, for scenarios with tracker: mpc
```

A display server (or `xvfb-run`) is needed for every mode, including stills:
the frame comes out of an OpenGL context.

A display server (or virtual framebuffer such as `xvfb`) is required for
interactive use. For headless recording, `xvfb-run` and `ffmpeg` are needed.

### Usage

```bash
# Interactive simulation (requires pygame)
arcosim map/city.yml

# Record to MP4 (requires pygame + ffmpeg)
arcosim map/city.yml -o output/city.mp4

# Limit recording duration
arcosim map/city.yml -o output/city.mp4 --record-duration 30

# Fast headless recording: skip animated planner-tree reveal
arcosim map/city.yml -o output/city.mp4 -d 60 --fast-record

# Save one frame as PNG, headless, at a chosen size and seed
xvfb-run -a arcosim map/city.yml -o output/city.png \
    --still 300 --width 1920 --height 1080 --seed 7

# Search a run for a good instant: one PNG per listed frame
xvfb-run -a arcosim map/city.yml -o output/scan/city.png \
    --still 120,300,600,900 --width 1920 --height 1080 --seed 7
```

| Flag | Effect |
|---|---|
| `--still FRAMES` | Save the listed zero-based recorded frames as PNG. One frame uses `--output` verbatim; several add an `_fNNNNN` suffix. |
| `--width` / `--height` | Framebuffer size for the still. Defaults to the scenario's own recording size (1280 x 720, or 1280 x 800 for ppp and rrp). |
| `--seed` | Pin unseeded planner sampling so the run repeats exactly. Seeds already set in the scenario YAML, such as the city world's `seed: 42`, are left alone. |

`--still` changes the frame sink and the framebuffer size and nothing else.
Without it every default is what it was, so `scripts/generate_videos.sh` and
the release workflow produce the same videos as before.

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
