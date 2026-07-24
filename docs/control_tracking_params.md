# Post–path-planning tracking parameters

After a global planner (RRT\*, SST, A\*, …) emits a waypoint polyline, ARCO
**does not replan** inside the online tracker.  The shared controller only
follows that reference with vehicle dynamics, soft obstacle barriers, and
(for MPC) a contouring cost.  Colour in the city race identifies which
planner produced the path; **all three racers use the same tracking /
control pipeline**.

This note lists every knob that sits **after** path planning: where it
lives, what it does, and how it tends to affect behaviour.  For the NMPC
math, see [control_mpcc.md](control_mpcc.md).  For Pure Pursuit / APF
usage, see [GUIDANCE.md](GUIDANCE.md).

---

## Pipeline placement

```
planner waypoints  →  ReferencePath (κ, s)  →  online tracker (MPC or PP)
                              ↑                        ↑
                     curvature floors / preview   weights, horizon, dynamics
```

| Layer | Role | Typical sources |
|-------|------|-----------------|
| Reference shaping | Arc-length path, curvature for `v_curve` | `ReferencePath` constants |
| Tracker selection | MPC vs Pure Pursuit (+ APF) | `simulator.tracker` in scenario YAML |
| NMPC cost / horizon | Contouring, lag, obstacles, prediction | `simulator.mpc.*`, `config/mpc.yml` |
| Vehicle limits | `v`, `ω`, `a`, `ω̇` bounds | `city_race_style.make_city_vehicle_config()` / `VehicleConfig` |
| Occupancy clearance | Soft barrier radius / planner clearance | `world.obstacle_clearance` |

Optional **trajectory optimization** (`TrajectoryOptimizer`) can sit between
planning and tracking in other pipelines; its weights live in
`tools/config/optimizer.yml` and are documented in
[planning_optimizer.md](planning_optimizer.md).  The city race currently
feeds planner polylines **directly** into the online tracker.

---

## 1. Tracker mode

| Parameter | Where | Effect |
|-----------|--------|--------|
| `simulator.tracker` | Scenario YAML (`map/city.yml`, …) | `"mpc"` → CasADi contouring NMPC; `"pure_pursuit"` → PP + optional APF |

City defaults to `mpc`.  Switching tracker changes the whole online law;
weights below apply only when `tracker: mpc`.

---

## 2. Prediction horizon (MPC)

YAML: `simulator.mpc.horizon.{step_count,dt}`  
Code defaults (city): `DEFAULT_CITY_HORIZON_STEP_COUNT`, `DEFAULT_CITY_HORIZON_DT`  
Global fallback: `config/mpc.yml` → `horizon`

| Key | Unit | Meaning |
|-----|------|---------|
| `step_count` | — | Number of prediction steps \(N\) |
| `dt` | s | Model step \(\Delta t\).  Horizon duration = \(N \cdot \Delta t\) |

**City today:** \(72 \times 0.05\,\mathrm{s} = 3.6\,\mathrm{s}\) (~half a block at
soft cruise).

| Raise horizon | Lower horizon |
|---------------|---------------|
| Earlier braking / corner open; sees obstacles farther ahead | Faster IPOPT; less anticipation; late reactions at kinks |
| Larger NLP → slower / more fragile solves | May cut corners the long horizon would have planned around |

Tune **duration** (\(N\cdot\Delta t\)) first; keep \(\Delta t\) near the
control period unless you have a reason to change discretization.

---

## 3. Contouring cost weights (MPC)

YAML: `simulator.mpc.weights.*`  
Loaded via `path_following_mpc_config_from_simulator` →
`PathFollowingMPCConfig`  
Globals: `config/mpc.yml` → `weights` (city overrides only listed keys)

| YAML key | Config field | Role |
|----------|--------------|------|
| `contour` | `weight_contour` | Quadratic penalty on **excess** lateral error outside the deadzone |
| `contour_deadzone` | `contour_deadzone` | Free band \(\lvert e_{\mathrm{lat}}\rvert \le d_{\mathrm{dz}}\) (m); cost is zero inside |
| `heading` | `weight_heading` | Heading error vs path tangent |
| `progress` | `weight_progress` | Track `v_ref` (cruise capped by curve speed) |
| `lag` | `weight_lag` | Penalty for falling behind virtual progress \(s_0 + v_{\mathrm{cruise}} t\) |
| `control` | `weight_control` | Effort on \((a, \dot\omega)\) — smoothness vs agility |
| `obstacle` | `weight_obstacle` | Soft directional obstacle barrier |
| `terminal` | `weight_terminal` | Terminal contour / heading (usually from `mpc.yml`) |
| `slack` | `weight_slack` | Soft-constraint slack (usually from `mpc.yml`) |

Barrier shape (not a YAML weight, but couples to `obstacle`):

| Key | Where | Role |
|-----|--------|------|
| `obstacle_barrier.power` | `config/mpc.yml` | Exponent \(p\) on the clearance violation term |

### How weights interact (rules of thumb)

- **`control` high + `ω̇` low** → car cannot turn into planner kinks →
  understeer, then overshoot when contour finally wins → weave / wall hits.
- **`contour` high + `contour_deadzone` tiny** → hunts the polyline; on
  jagged A\* paths this looks like zigzag.
- **`heading` high** on discontinuous planner yaw → left/right chasing.
- **`lag` / `progress` high** → prefers advancing \(s\) and holding
  `v_ref`; helps progress-first widening, but can pull through a soft
  barrier if the reference itself cuts a corner.
- **`obstacle` high** strengthens soft barriers only; it is **not** a hard
  road tube.  Sparse point-cloud samples can still miss a face or fight
  contouring.

City values are scenario YAML; global `mpc.yml` stays classic stiff
(`contour_deadzone: 0`, `lag: 0`) for non-city demos.

---

## 4. Vehicle / actuator limits (shared)

City: `arco.simulator.sim.city_race_style` → `make_city_vehicle_config()`  
Wired into both the simulated `DubinsVehicle` and `DubinsVehicleLimits`
for the MPC.

| Constant / field | Unit | Meaning |
|------------------|------|---------|
| `CITY_MAX_SPEED` / `max_speed` | m/s | Forward speed ceiling |
| `min_speed` | m/s | Floor (city: `0`) |
| `CITY_CRUISE_SPEED` / `cruise_speed` | m/s | Nominal `v_ref` on straights; also seeds MPC cruise |
| `CITY_MAX_TURN_RATE_DEG` / `max_turn_rate` | deg/s → rad/s | \(\lvert\omega\rvert\) max → sets \(R_{\min}=v/\omega\) and `v_curve` |
| `CITY_MAX_TURN_RATE_DOT_DEG` / `max_turn_rate_dot` | deg/s² → rad/s² | \(\lvert\dot\omega\rvert\) max — how fast steering can change |
| `CITY_MAX_ACCELERATION` / `max_acceleration` | m/s² | \(\lvert a\rvert\) bound on speed changes |
| `CITY_LOOKAHEAD_DISTANCE` / `lookahead_distance` | m | PP carrot distance; MPC uses related look-ahead for path / obstacle preview |
| `CITY_GOAL_RADIUS` / `goal_radius` | m | Goal acceptance radius |
| `curvature_gain` | 1/m | PP feed-forward (city MPC sets `0`) |

**Lane geometry budget (docs / tests, not an NLP constraint):**
`CITY_ROAD_HALF_WIDTH` (15 m) — use when judging whether a corner radius
from \(\omega\) and `v_curve` fits the navigable lane.

---

## 5. Reference-path curvature (post-plan shaping)

Implemented on `ReferencePath` (`arco.control.mpc.reference_path`).
Affects **curve-limited speed** \(v_{\mathrm{curve}}=\omega_{\max}/\lvert\kappa\rvert\)
inside the MPC — not the planner.

| Constant | Unit | Meaning |
|----------|------|---------|
| `_CURVATURE_DS_CAP_M` | m | Max arc length used to spread a heading turn into \(\kappa\) |
| `_CURVATURE_DS_FLOOR_M` | m | Min spread for non-trivial turns (avoids Dirac \(\kappa\) on ~1 m A\* stubs → IPOPT failure / stuck \(v=0\)) |
| `_CURVATURE_ABS_MAX` | 1/m | Hard cap on \(\lvert\kappa\rvert\) after preview |
| `_CURVATURE_PREVIEW_DS_M` | m | Backward visibility of an upcoming corner \(\kappa\) so braking starts before the kink |

These are code constants (not scenario YAML).  Changing them retunes
braking aggressiveness and NLP solvability for dense polylines.

---

## 6. Soft obstacle sampling (MPC implementation)

Not YAML today; class attributes / logic on `DubinsPathFollowingMPC`:

| Item | Role |
|------|------|
| `_OBSTACLE_SAMPLE_COUNT` | Max nearest-obstacle samples fed into the NLP (city: 5) |
| Query centers | Vehicle pose + path preview along look-ahead (current default) |
| Occupancy `clearance` | Usually `world.obstacle_clearance` from the map YAML |

Optional forward / flank probes can catch cuts the centerline misses, but
increase barrier conflict with contouring when weights are already high —
prefer tuning weights before densifying samples.

---

## 7. Solver budget (MPC)

| Key | Where | Role |
|-----|--------|------|
| `solver.max_iter_count` | `config/mpc.yml` → `max_solver_iter_count` | IPOPT iteration cap |

UI strings such as `Optim: 0: CONVERGENCE: RELAT…` report **IPOPT return
status**, not “tracking quality”.  Relative convergence with a badly
shaped cost still yields drunk trajectories.

---

## 8. Pure Pursuit path (when `tracker: pure_pursuit`)

Same `VehicleConfig` limits, plus:

| Parameter | Role |
|-----------|------|
| `lookahead_distance` | Carrot distance along the path |
| `curvature_gain` | Feed-forward speed / steer shaping from path curvature |
| APF / occupancy | Optional repulsion from nearby obstacles in the tracking loop |

PP has no contouring deadzone / lag weights; obstacle behaviour is geometric
+ APF, not the NMPC barrier.

---

## 9. Map / world knobs that couple to tracking

Still “after planning” in the sense that they shape clearance the tracker
sees (and that the planner used):

| Key | Typical file | Coupling |
|-----|--------------|----------|
| `obstacle_clearance` | `map/city.yml` → `world` | Planner clearance and soft-barrier scale |
| `road_half_width` | city map | Lane width budget (visualization / feasibility judgment) |
| `obstacle_sampling_spacing` | city map | Density of building samples in the KD-tree cloud |

Release city videos may use `map/city_mpc_preview.yml` (shorter planner
budgets).  That changes **path quality**, not the tracker formula — but a
worse polyline stresses the same post-plan knobs harder.

---

## Suggested tuning order

Change **one family at a time**; keep the other racers on the same pipeline.

1. **Actuator authority** — `control`, `max_turn_rate`, `max_turn_rate_dot`,
   `max_acceleration` (can the car physically follow a corner?).
2. **Lane band** — `contour`, `contour_deadzone` (how hard to hug vs widen).
3. **Heading** — reduce if A\*/grid paths zigzag the steer command.
4. **Progress law** — `progress`, `lag` (advance \(s\) vs nail geometry).
5. **Soft obstacles** — `obstacle`, barrier power, sample probes
   (only after the car can stay near the topological lane).
6. **Horizon** — \(N\cdot\Delta t\) for anticipation vs solve cost.
7. **Curvature floors** — only if IPOPT fails or never brakes before kinks.

---

## City defaults vs aggressive stiffen (do not re-apply)

**Current city defaults** (restored lane-viable / progress-first column)
are shared by blue / green / purple.  The right-hand column is the
v0.3.5 over-stiff retune — kept only as an anti-pattern.

| Knob | Current (lane-viable) | Aggressive stiffen (avoid) |
|------|------------------------|----------------------------|
| `contour` | 8 | 18 |
| `heading` | 4 | 6 |
| `control` | 0.5 | 4 |
| `progress` | 4 | 3 |
| `lag` | 4 | 2 |
| `obstacle` | 120 | 280 |
| `contour_deadzone` | 2.5 m | 1.2 m |
| `CITY_MAX_TURN_RATE_DOT_DEG` | 90 | 55 |
| Obstacle NLP samples | pose + path preview (5) | + forward/flank probes (9) |
| Horizon | 72 × 0.05 s | same |

Raising `control` / obstacle / contour together while cutting \(\dot\omega\)
and the deadzone often yields **solver “convergence” with poor tracking**:
understeer into corners, then lateral hunting, then soft-barrier
penetration.  Nudge **one** knob at a time from the current column.

Keep the **κ floor / preview** (`ReferencePath`): that fix is about NLP
solvability on dense A\* stubs, not about stiffness.

---

## File cheat-sheet

| Concern | File |
|---------|------|
| City MPC horizon + weights | `map/city.yml`, `map/city_mpc_preview.yml` |
| Global MPC defaults | `src/arco/config/mpc.yml` |
| City vehicle limits | `src/arco/simulator/sim/city_race_style.py` |
| YAML → config wiring | `src/arco/simulator/sim/tracking.py` |
| Contouring NLP | `src/arco/control/mpc/path_following.py` |
| Path κ / progress | `src/arco/control/mpc/reference_path.py` |
| Math formulation | [control_mpcc.md](control_mpcc.md) |

---

*Update this document in the same PR when city tracking defaults or
`PathFollowingMPCConfig` fields change.*
