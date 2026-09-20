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

## Architecture & Runtime

In-process pipeline execution and message passing are implemented in Rust in the `arco-runtime` crate (`crates/arco-runtime/`), providing a thread-safe, bounded, typed message bus (`InMemoryBus`), lifecycle-managed nodes (`PipelineNode`), and the orchestrator (`PipelineRunner`). On the Python side, these are exposed via `arco.middleware` and `arco.pipeline`.

| Aspect | Current Architecture | Future / Distributed Options |
|--------|----------------------|------------------------------|
| Process model | In-process threaded nodes (`arco-runtime`) | Multi-process worker nodes |
| Messaging | Typed, bounded thread-safe bus (`Bus`) | IPC / network channels |
| Telemetry | Dedicated publisher channel (`TelemetryPublisher`) | Dashboard / streaming endpoints |
| Execution | Coordinated runner lifecycle (`PipelineRunner`) | Distributed workflow runners |

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
