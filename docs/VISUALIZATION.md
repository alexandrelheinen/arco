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
arcosim src/arco/tools/map/city.yml

# Record to MP4 (requires pygame + ffmpeg)
arcosim src/arco/tools/map/city.yml --record output/city.mp4

# Limit recording duration
arcosim src/arco/tools/map/city.yml --record output/city.mp4 --record-duration 30

# Static image mode — opens matplotlib window
arcosim src/arco/tools/map/city.yml --image

# Static image mode — save to file (headless-safe)
arcosim src/arco/tools/map/city.yml --image --record output/city.png

# --static is an alias for --image
arcosim src/arco/tools/map/city.yml --static --record output/city.png
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
```
