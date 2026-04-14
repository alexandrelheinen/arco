# ARCO Pipeline Architecture

## Design Principle

Each step of the ARCO processing pipeline is designed to run as an **independent
OS process**.  Every step:

1. **Reads** one or more input files (config, occupancy map, planned path, …).
2. **Executes** its algorithm (mapping, planning, optimization, simulation).
3. **Writes** one or more output files (occupancy JSON, path JSON, trajectory
   JSON, video, metrics JSON).

This "read file A → write file B" discipline means:

- Steps can be restarted independently without re-running the whole pipeline.
- Steps can run on different machines (distributed pipeline).
- Telemetry / monitoring can be injected between any two steps.
- Unit-testing a step requires only providing input files and checking output
  files.

## Current State vs. Target State

| Aspect | Current | Target |
|--------|---------|--------|
| Process model | Single Python process (in-memory) | One OS process per step |
| IPC | Python function calls / shared objects | File I/O + pub/sub middleware |
| Telemetry | JSON temp-file polling | Proper pub/sub (see [ROADMAP.md](ROADMAP.md) §IPC) |
| Restart | Full re-run required | Per-step restart from last written file |

## Pipeline Steps

```
Step 1 — Mapping
  reads:  config.yml, obstacles definition
  writes: occupancy.json (KDTree obstacle point set + clearance)

Step 2 — Planning
  reads:  occupancy.json, planner config (bounds, step_size, …)
  writes: path.json (list of waypoints in C-space)

Step 3 — Trajectory Optimization
  reads:  path.json, occupancy.json, vehicle config
  writes: trajectory.json (time-stamped waypoints + durations)

Step 4 — Simulation / Recording
  reads:  trajectory.json, scene config
  writes: output/ (video.mp4, metrics.json, images/)
```

## Telemetry Side-Channel

Each step writes live metrics to a telemetry channel (currently a JSON temp
file, see `arco.planning.continuous.telemetry`).  The loading screen (and
future dashboards) poll this channel to display live stop-criteria progress.

<!-- TODO(PROCESS-PIPELINE): Implement the file-based step runner and CLI
     entry points so each step above can be invoked as:
       python -m arco.pipeline.step_mapping  config.yml occupancy.json
       python -m arco.pipeline.step_planning occupancy.json path.json
     This will replace the current in-process scene.build() approach. -->
