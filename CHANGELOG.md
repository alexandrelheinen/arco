# Changelog

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- The images a published release attaches are the seven gallery plates
  (`field`, `wavefront`, `growth`, `contest`, `refine`, `pursuit`,
  `reachability`) on the nocturne and atlas grounds, plus the mark and
  `field-og.png`. They keep the distance field, the trees and the
  routes, and drop the title, the legend, the side chart and the wide
  bloom. The flat gesture set shipped in 0.6.0 is no longer rendered.

## [0.6.0] - 2026-09-23

A published release now carries a quiet image set for the web. Each file
is one gesture and contains no title, number, or legend. The seven deck
plates are unchanged.

### Added

- `tools/render_web.py` renders `mark`, `arc`, `branch`, `flood`,
  `pair`, `track`, and `ribbon` on a dark ground and a light ground,
  plus `arc-og.png`, a 1200×630 crop of the dark cover.
- Publishing a release renders that set and attaches the PNG files.
  The simulator videos are attached by the same workflow, as before.

## [0.5.0] - 2026-09-18

The library is compiled. Every planner, controller, map and guidance
routine outside `arco.simulator` now runs as Rust in an extension module,
and the Python package re-exports it.

### For a caller who does nothing

Imports do not move. `from arco.planning import RRTPlanner` resolves the
same name from the same module path it always did, argument names and
default values are unchanged, and the exception a call raises keeps the
type it had. Scripts written against 0.4 run unmodified.

### Added

- Wheels carrying a stable ABI, one per platform for Python 3.10 and
  above, so installing needs no Rust toolchain. A source distribution
  covers platforms without a wheel and does need one.
- `MPCStepResult.solver_status` distinguishes an infeasible program from
  an exhausted budget from a numerical failure, where the previous
  release reported one `solve_failed` for all three.

### Changed

- Planners release the interpreter lock while they search, so a caller
  running independent problems through its own thread pool is no longer
  blocked by them. Passing a Python callable to a policy hook
  (`sampler=`, `steerer=`, `segment_free=`, `cost=`, and the rest) puts
  the interpreter back in the inner loop and gives up that speedup, which
  is why each hook also accepts a built-in policy by name.
- The two predictive controllers solve a sequence of convex programs
  rather than one nonlinear program. They reach a different and also valid
  solution for the same input, and `MPCStepResult.cost` reports the
  surrogate objective, which is comparable across steps of one controller
  and not against a number 0.4 printed.
- `obstacle_barrier_power` is accepted and no longer shapes the obstacle
  barrier, which is a half-space with a penalized slack in this release.
- Every controller validates the elapsed interval it is given and refuses
  one that disagrees with its configured model step, where 0.4 accepted
  the argument and quietly used the configured value.
- Commands leaving the control layer pass a saturation and a rate limit.
  Both default to infinity, so a caller that never configured a limit gets
  what it asked for.

### Removed

- The `mpc` extra, and with it the CasADi and IPOPT dependency. Nothing in
  the package imports CasADi, and `pip install arco[mpc]` no longer
  resolves.

### Known limitations

- A joint-space controller does not steer around an obstacle whose shape
  the occupancy protocol cannot describe, because the barrier is a ball
  around the nearest reported point. Deviation A-34 has the detail.
- A path-following controller stops short of an obstacle sitting on its
  reference rather than going around it. Deviation A-35.
- Running plans across threads buys wall-clock time on an ordinary Linux
  kernel and does not under WSL2, where four concurrent plans each take
  about twelve times as long as one alone. `docs/decisions.md` records the
  measurements.

Every behavior difference between this release and 0.4 carries an entry in
[docs/rust/DEVIATIONS.md](docs/rust/DEVIATIONS.md) naming the symbol, the
old behavior, the new behavior and the reason.
