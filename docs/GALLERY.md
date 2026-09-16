# Illustration gallery

Seven 16:9 plates, in two themes, rendered from real runs of the shipped
planners and controllers. They exist to illustrate the library in places
where a screen recording does not fit: a pitch deck, a project page, a
paper figure, a release header.

Every curve on every plate is solver output. The trees come from
`RRTPlanner.get_tree` / `SSTPlanner.get_tree`, the wavefront from
`AStarPlanner.plan_with_diagnostics`, the smooth trajectory from
`TrajectoryPruner` plus `TrajectoryOptimizer`, the executed motion from
`TrackingLoop` driving a `DubinsVehicle`, and the reachable set from
forward integration of that same vehicle model. Nothing is traced by
hand and nothing is mocked.

## Regenerating

```bash
pip install -e ".[dev]"
python3 tools/render_gallery.py
```

No display server is required: the gallery is a matplotlib renderer, not
`arcosim`. The first run is slow because it solves everything (the
8 000-sample RRT* alone takes about 100 s); results are pickled under
`tools/output/gallery_cache/`, so every later run is a few seconds.

| Flag | Effect |
|---|---|
| `--theme nocturne\|atlas` | Render one theme; repeatable. Defaults to both. |
| `--plate 01_field` | Render one plate; repeatable. Defaults to all. |
| `--width` / `--dpi` | Figure width in inches and resolution. The committed set is `16 x 120` (1920 x 1080); `--dpi 240` gives 3840 x 2160 for print. |
| `--output DIR` | Destination; files land in `DIR/<theme>/<plate>.png`. |

Delete `tools/output/gallery_cache/` to force a re-solve, for example
after changing `tools/illustration/world.py`.

## Themes

| Theme | Use |
|---|---|
| `nocturne` | Deep-navy stage, neon bloom. Slides, web, dark READMEs. |
| `atlas` | Warm paper stage, ink strokes. Print and journal figures. |

Algorithm hues are read from `src/arco/config/colors.yml` through
`arco.config.palette`, so the gallery and the `arcosim` renderer stay the
same family of blue (RRT*), green (SST) and violet (A*).

## The plates

| Plate | Shows | Comes from |
|---|---|---|
| `01_field` | The cover: a dense RRT* tree coloured by cost-to-come, its solution, and the optimised trajectory as a speed ribbon. | `RRTPlanner`, `TrajectoryPruner`, `TrajectoryOptimizer` |
| `02_wavefront` | A* flooding the grid — tint is expansion order, contours are equal-effort frontiers. | `AStarPlanner.plan_with_diagnostics` |
| `03_growth` | The same seeded RRT* at 1 200 / 3 000 / 8 000 samples, with the route length it returns plotted underneath. | `RRTPlanner` |
| `04_contest` | A*, RRT* and SST on one map, with their exploration behind their answers. | all three planners |
| `05_refine` | A corner at full zoom: raw hops, pruned shortcuts, optimised curve, plus the speed profile. | `TrajectoryPruner`, `TrajectoryOptimizer` |
| `06_pursuit` | Closed-loop tracking — executed line coloured by signed lateral error, with the lookahead rays the controller aimed at. | `TrackingLoop`, `PurePursuitController`, `ArtificialPotentialField`, `DubinsVehicle` |
| `07_reachability` | Turn-rate ramps integrated through the vehicle model, split into what the map admits and what it removes. | `DubinsVehicle.step`, `KDTreeOccupancy` |

## The scene

`tools/illustration/world.py` builds one purpose-made 160 x 90 basin of
organic obstacle bodies rather than reusing a scenario from `map/`. The
scenarios there are built for `arcosim` and read as engineering; a field
of smooth bodies gives the planners something to curve around, which is
what makes a plate worth looking at. It is still an ordinary
`KDTreeOccupancy` and an ordinary `EuclideanGrid`, planned by the shipped
planners with no special cases.

## Code layout

```text
tools/render_gallery.py            CLI entry point
tools/illustration/
├── theme.py        colours, type sizes, the two themes
├── ramps.py        colour ramps for cost, expansion order, speed, error
├── canvas.py       the 16:9 figure, bloom/ribbon/tree primitives, chrome
├── stage.py        the world panel every plate is built on
├── world.py        the bespoke scene
├── solution.py     every solver run, with a pickle cache
└── plates/         one module per plate
```

Adding a plate means adding a module under `plates/` that exposes
`TITLE`, `SUBTITLE` and `render(theme, world, solution, output, width_in,
dpi)`, then listing it in `plates/__init__.py`.
