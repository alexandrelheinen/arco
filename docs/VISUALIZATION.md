# Visualization Tools

ARCO ships two visualization tools: **arcoex** for static image generation and
**arcosim** for real-time interactive simulation. Both tools are driven by YAML
scenario files and are distributed as optional extras.

## arcoex — Static Example Runner

`arcoex` runs a scenario through the ARCO pipeline and saves the result as a
static matplotlib figure. It is the tool used by the `generate-images` CI gate.

### What it does

- Loads a scenario YAML file and dispatches to the matching example module in
  `arco.tools.examples`.
- Runs the full mapping → planning → guidance pipeline for the scenario.
- Renders the result as a matplotlib figure (interactive window or saved file).

### Dependencies

```bash
pip install arco[tools]   # matplotlib + pyyaml
```

No display server is needed when saving to a file (`--save`).

### Usage

```bash
# Interactive window
arcoex src/arco/tools/map/city.yml

# Save to file (headless-safe)
arcoex src/arco/tools/map/city.yml --save output/city.png
```

### Supported scenarios

| Scenario | Description |
|----------|-------------|
| `astar`  | A* grid search with obstacle map |
| `city`   | Route planning on the city road network |
| `occ`    | Object-centric control in obstacle field |
| `ppp`    | PPP robot arm kinematics |
| `rr`     | RR robot arm kinematics |
| `rrp`    | RRP robot arm kinematics |
| `vehicle`| Vehicle trajectory with tracking controller |

The `scenario:` key in the YAML header selects which example to run.

---

## arcosim — Real-Time Simulation

`arcosim` runs a scenario as an interactive real-time simulation using pygame
and PyOpenGL. It can also record to an MP4 file.

### What it does

- Loads a scenario YAML file and dispatches to the matching simulator module in
  `arco.tools.simulator.main`.
- Opens a pygame window that animates the scenario in real time.
- Optionally records the session to an MP4 file (requires ffmpeg).

### Dependencies

```bash
pip install arco[tools,pygame]   # adds pygame >= 2.0 and PyOpenGL >= 3.1
```

A display server (or virtual framebuffer such as `xvfb`) is required for
interactive use. For headless recording, `xvfb-run` and `ffmpeg` are needed.

### Usage

```bash
# Interactive simulation
arcosim src/arco/tools/map/city.yml

# Record to MP4 (90-second maximum by default)
arcosim src/arco/tools/map/city.yml --record output/city.mp4

# Limit recording duration
arcosim src/arco/tools/map/city.yml --record output/city.mp4 --record-duration 30
```

Scenario-specific flags (e.g. `--fps`, `--dt`, `--camera`) can be appended and
are forwarded verbatim to the underlying simulator.

### Supported scenarios

The same set as arcoex: `astar`, `city`, `occ`, `ppp`, `rr`, `rrp`, `vehicle`.

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

The YAML files for the built-in scenarios live in `src/arco/tools/map/`.

---

## Running the full CI validation locally

The `scripts/pre_push.sh` script exercises both tools as part of the local CI
gate:

```bash
# Full validation (requires xvfb + ffmpeg for smoke/video gates)
bash scripts/pre_push.sh

# Skip pygame-dependent gates on headless machines
bash scripts/pre_push.sh --no-smoke --no-videos
```
