# Contouring NMPC (MPCC path following)

ARCO's SE(2) online tracker `DubinsPathFollowingMPC` is a **nonlinear model
predictive contouring controller** (MPCC): it augments the vehicle state
with a path parameter \(s\) driven by its own **virtual progress speed**
decision variable, splits the position error into **contouring** (lateral)
and **lag** (longitudinal) components, and solves the resulting nonlinear
program each control step under Dubins / unicycle dynamics, as a short
sequence of convex quadratic programs linearized about the previous
solution and solved by Clarabel (ADR-002 in [decisions.md](decisions.md);
deviation A-02 in [rust/DEVIATIONS.md](rust/DEVIATIONS.md)).

This is the classical Lam / Liniger MPCC structure, carried through a
sequential quadratic programming solve rather than a single interior-point
one. This note documents **exactly what ARCO implements**, including the
design decisions that fix the historical city-race failure modes (zigzag,
junction orbits, parked stalls at sharp kinks), and the points where the
convex reformulation changes what the controller reports or does with a
configuration knob.

## Key references (family)

- Lam, D., Manzie, C., & Good, M. (2010). Model Predictive Contouring Control.
- Liniger, A., Domahidi, A., & Morari, M. (2015). Optimization-based
  autonomous racing of 1:43 scale RC cars (MPCC).
- Romero, A., et al. (2021). Model Predictive Contouring Control for
  Time-Optimal Quadrotor Flight.
- Ji, T., et al. (2020). CMPCC: Corridor-based Model Predictive Contouring
  Control for Aggressive Drone Flight (hard tubes).

## Implementation status

Fully implemented in:

| Piece | File |
|-------|------|
| Sequential convex program / step | `crates/arco-control/src/mpc/path_following.rs` |
| Cost terms, obstacle barrier | `crates/arco-control/src/mpc/costs.rs`, `crates/arco-control/src/mpc/model.rs` |
| QP assembly and solve | `crates/arco-control/src/mpc/qp.rs` |
| Arc-length path, \(\kappa\), projection | `crates/arco-control/src/mpc/reference.rs` |
| Python binding | `crates/arco-py/src/mpc/path.rs`, `crates/arco-py/src/mpc/reference.rs` |
| Python re-export shim | `src/arco/control/mpc/path_following.py`, `src/arco/control/mpc/reference_path.py` |
| Metrics loop | `src/arco/control/mpc/tracking_loop.py` (binding: `crates/arco-py/src/mpc/tracking.rs`) |
| City / sim factory | `src/arco/simulator/sim/tracking.py` |
| Closed-loop city report | `tools/city_tracking_report.py` |
| Demo (stiff vs lane-aware) | `tools/mpc_progress_first_demo.py` |

`DubinsPathFollowingMPC` is a compiled extension type reached through
`arco._arco`; it carries no optional dependency.

---

## 1. Reference path

Global planners (RRT\*, SST, A\*) return an ordered waypoint polyline
\(\{(x_i, y_i)\}\).  `ReferencePath` builds an arc-length parameterization

\[
p(s) = \bigl(x_{\mathrm{ref}}(s),\, y_{\mathrm{ref}}(s)\bigr),\qquad
s \in [0, L],
\]

with heading \(\psi_{\mathrm{ref}}(s)\) from segment tangents and an
approximate curvature \(\kappa(s)\) used for progress-speed capping.

Each control step looks the reference up directly on this polyline:
position **linearly interpolated** along the current segment, and
curvature linearly interpolated between the per-vertex values the next
subsection derives. The nonlinear formulation this replaced instead
resampled the reference onto a uniform arc-length grid (2 m resolution,
200-1500 samples) and read it back through a cubic B-spline interpolant,
because that solver built one symbolic nonlinear graph per control step
and a piecewise-linear lookup's discontinuous gradient at a polyline kink
could stall it (`Maximum_Iterations_Exceeded`) exactly where tracking is
hardest. The sequential convex solve re-linearizes from each iterate's own
nominal position instead of holding one global symbolic graph, so a
piecewise-linear lookup costs nothing in convergence, and both the grid
and the spline are gone.

### Runway extension

`set_reference` appends one prediction-horizon length of straight
"runway" along the final tangent.  Without it, the progress bounds pinch
\(S\) against the arc-length cap near the goal and the program fails on
the last meters of the race.

### Curvature estimate

At each interior vertex, the turn between consecutive headings is

\[
\Delta\psi_i = \mathrm{wrap}\bigl(\psi_i - \psi_{i-1}\bigr),\qquad
\Delta s_i = \min(\Delta s_{\mathrm{in}},\,\Delta s_{\mathrm{out}},\, s_{\mathrm{cap}}),
\]

and for non-trivial turns \(\Delta s_i \leftarrow \max(\Delta s_i,\, s_{\mathrm{floor}})\)
with \(s_{\mathrm{cap}} = 20\,\mathrm{m}\) and \(s_{\mathrm{floor}} = 8\,\mathrm{m}\),
then \(\kappa_i = \Delta\psi_i / \Delta s_i\).  A **short** backward
max-preview (12 m) keeps a corner \(\kappa\) visible just before the
vertex; longer-range braking is planned by the receding horizon itself
(a long preview double-counts the conservatism and drags cruise down on
every straight).  \(|\kappa|\) is clipped to
\(\kappa_{\max} = 0.35\,\mathrm{m}^{-1}\) so short A\*/optimizer stubs
cannot create Dirac \(\kappa\).

---

## 2. Prediction model (Dubins / unicycle)

Decision variables over horizon \(N\) with step \(\Delta t\):

| Symbol | Meaning |
|--------|---------|
| \(X_k = (p_x, p_y, \psi, v, \omega)_k\) | pose, speed, yaw rate |
| \(U_k = (a, \dot\omega)_k\) | accel, yaw acceleration |
| \(s_k\) | path parameter (arc length) |
| \(v_{s,k}\) | **virtual progress speed** \(\dot s\) (decision variable) |

Discrete Euler dynamics (matching `DubinsVehicle.step` saturation
semantics):

\[
\begin{aligned}
p_{x,k+1} &= p_{x,k} + v_k \cos\psi_k\,\Delta t,\\
p_{y,k+1} &= p_{y,k} + v_k \sin\psi_k\,\Delta t,\\
\psi_{k+1} &= \psi_k + \omega_k\,\Delta t,\\
v_{k+1} &= v_k + a_k\,\Delta t,\\
\omega_{k+1} &= \omega_k + \dot\omega_k\,\Delta t,\\
s_{k+1} &= s_k + v_{s,k}\,\Delta t.
\end{aligned}
\]

Box constraints: \(v \in [v_{\min}, v_{\max}]\),
\(|\omega| \le \omega_{\max}\), \(|a| \le a_{\max}\),
\(|\dot\omega| \le \dot\omega_{\max}\), \(s_k \in [0, L]\), and
\(v_{s,k} \ge 0\) (progress never reverses).

### The model \(\Delta t\) must equal the control period

The first predicted state \((v_1, \omega_1)\) is the command target the
plant rate-limits toward for one **control period**.  If the model
\(\Delta t\) is shorter than the control period (the historical city
setup: 0.05 s model vs 0.1 s simulator step), the plant travels twice as
far per tick as the plan's first step — a structural source of
closed-loop zigzag.  City wiring: \(50 \times 0.1\,\mathrm{s} = 5.0\,\mathrm{s}\)
of preview, longer than the 4.8 s full-stop braking time from cruise.

---

## 3. Contouring / lag errors

At predicted progress \(s_k\), interpolate
\((x_{\mathrm{ref}}, y_{\mathrm{ref}}, \psi_{\mathrm{ref}}, \kappa)\) and
split the position error in the path frame:

\[
\begin{aligned}
e_{c,k} &=
-(p_{x,k}-x_{\mathrm{ref}})\sin\psi_{\mathrm{ref}}
+(p_{y,k}-y_{\mathrm{ref}})\cos\psi_{\mathrm{ref}}
&&\text{(contouring, lateral)}\\
e_{l,k} &=
(p_{x,k}-x_{\mathrm{ref}})\cos\psi_{\mathrm{ref}}
+(p_{y,k}-y_{\mathrm{ref}})\sin\psi_{\mathrm{ref}}
&&\text{(lag, longitudinal)}
\end{aligned}
\]

The **lag error is structural**: it is the only term coupling the virtual
progress \(s\) to the vehicle, so `weight_lag` must be strictly positive
(enforced at construction).  No projection heuristics or monotonicity
constraints are needed *inside* the program.

**Heading error** \(e_{\psi,k} = \psi_k - \psi_{\mathrm{ref}}(s_k)\) uses a
smooth \(2\pi\)-periodic surrogate
\(\ell_\psi(e_\psi) = \sin^2 e_\psi + (1-\cos e_\psi)^2\), kept at a
**small weight**: heavy heading tracking on kinked planner references
fights the contour/lag pair and produces steering chatter.

**Deadzone (optional free band).**  With `contour_deadzone` \(> 0\), only
the excess \(\max(|e_c| - d_{\mathrm{dz}}, 0)\) is penalized.  The default
\(d_{\mathrm{dz}} = 0\) uses the plain smooth quadratic \(e_c^2\) (no
`fabs` kink at zero).

---

## 4. Progress law: linear reward + curve-limited cap

The progress incentive is **linear** (classical MPCC):

\[
J_{\mathrm{prog}} = -\,w_p \sum_{k} v_{s,k}\,\Delta t
\;=\; -\,w_p\,(s_N - s_0),
\]

with the virtual speed hard-capped by the curve-limited reference speed,
written as two smooth inequalities (no `fmin`/`fabs` kinks in the
constraint set):

\[
v_{s,k} \le v_{\mathrm{cruise}},
\qquad
v_{s,k}\,\sqrt{\kappa(s_k)^2 + \varepsilon} \le \omega_{\max}
\;\;\Leftrightarrow\;\;
v_{s,k} \le \frac{\omega_{\max}}{|\kappa(s_k)|}.
\]

**Why not quadratic speed-matching?**  The previous cost
\(w_v (v_s - v_{\mathrm{ref}}(s_k))^2\) is a trap: at a sharp kink
\(v_{\mathrm{ref}}(s)\) is small, so *parking at the kink* costs almost
nothing while accelerating away looks expensive over the horizon — the
solver then converges to a permanent full stop (the city A\* racer
stall).  A
linear reward makes advancement pay everywhere; corner braking stays
feed-forward through the \(v_s\) cap, which the lag term transfers to the
actual vehicle speed.

**Measurement / seed.**  Each `step()` projects the measured pose onto the
path in a local window around the current \(s\) (a global nearest-point
search can flip to another road corridor at junctions) and never rewinds:
\(s \leftarrow \max(s, s_{\mathrm{proj}})\).  Recovery arcs catch up to
\(s\) through the lag cost instead of resetting it.

**Warm start / anti-stall initialization.**  Each control step linearizes
about a **nominal** trajectory before solving, and that nominal warm-starts
from the shifted previous solution while that solution keeps moving. If
the warm start advances less than
\(\max(1, 0.1\, v_{\mathrm{cruise}} N \Delta t)\) meters over the horizon (a
"parked" solution), and the vehicle is not already floored at its minimum
speed or within one horizon of the goal, the nominal is rebuilt as a
**rollout**: the model integrated forward with a feed-forward acceleration
toward the curve-limited cruise speed and a turn rate toward the reference
heading, both inside their own limits. Integrating the model, rather than
pinning every predicted pose onto the path the way the earlier nonlinear
implementation did, is what keeps the rollout inside the trust region the
sequential solve linearizes around (see [Online problem](#6-online-problem-and-solver)):
a nominal the vehicle could not reach in one step can sit further from
every point the dynamics can actually produce than the trust radius
allows, leaving no feasible point near it. Seeding inside the moving basin
is what lets the sequential solve escape the parked local minimum at sharp
kinks.

---

## 5. Stage cost (what the program minimizes)

For \(k = 0,\ldots,N-1\):

\[
\begin{aligned}
J &=
\sum_{k=0}^{N-1}
\Big[
w_c\,e_{c,k}^2
+ w_l\,e_{l,k}^2
+ w_\psi\,\ell_\psi(e_{\psi,k})
- w_p\,v_{s,k}\,\Delta t
+ w_u\,(a_k^2+\dot\omega_k^2)
+ J_{\mathrm{obs},k}
\Big] \\
&\quad
+ w_T\bigl(e_{c,N}^2 + \ell_\psi(e_{\psi,N})\bigr)
+ w_l\,e_{l,N}^2,
\end{aligned}
\]

where \(e_{c}^2\) becomes \(\max(|e_c|-d_{\mathrm{dz}},0)^2\) when a
deadzone is configured. This is the cost the nonlinear problem states.
Each sequential iterate minimizes a convex surrogate of it instead: the
contouring, lag and heading terms become the square of the affine
expansion of \(e_c\), \(e_l\) and \(e_\psi\) about the nominal trajectory
(dropping the second-order term is what keeps each block positive
semidefinite), the deadzone above becomes an exact epigraph reformulation
with a slack and two linear rows rather than an approximation of the
\(\max\) itself, and \(J_{\mathrm{obs},k}\) takes the different form
[Soft obstacle barriers](#soft-obstacle-barriers) below describes.
`MPCStepResult.cost` reports the value of that convex surrogate at the
solution, not the nonlinear \(J\) above: it is comparable across steps of
one controller and not against a number the earlier nonlinear
implementation printed (deviation A-31).

Weights map to `PathFollowingMPCConfig` / YAML `simulator.mpc.weights`:

| Weight | Config / YAML key |
|--------|-------------------|
| \(w_c\) | `weight_contour` / `contour` |
| \(w_l\) | `weight_lag` / `lag` (must be \(> 0\)) |
| \(w_\psi\) | `weight_heading` / `heading` |
| \(w_p\) | `weight_progress` / `progress` |
| \(w_u\) | `weight_control` / `control` |
| \(w_{\mathrm{obs}}\) | `weight_obstacle` / `obstacle` |
| \(w_T\) | `weight_terminal` / `terminal` |
| \(d_{\mathrm{dz}}\) | `contour_deadzone` |

### Soft obstacle barriers

The nonlinear cost above penalized penetration of the clearance margin
with the power-law, forward-cone-weighted term

\[
J_{\mathrm{obs},k}
=
\sum_j
w_{\mathrm{obs}}\,
\Bigl(\max\bigl(\tfrac{c - d_{k,j}}{c},\,0\bigr)\Bigr)^{p}
\Bigl(0.2 + 0.8\max\bigl(\tfrac{\cos\psi_k\,\Delta x_{k,j} + \sin\psi_k\,\Delta y_{k,j}}{d_{k,j}},\,0\bigr)\Bigr),
\]

with clearance \(c\), obstacle offset \((\Delta x, \Delta y)\), distance
\(d_{k,j}\), and power \(p =\) `obstacle_barrier_power`. That expression
still exists, as the float-only `obstacle_barrier` and
`forward_cone_factor` helpers in `arco.control.mpc.costs`, but the
controller's own program no longer evaluates it (deviation A-30).

The keep-out region around an obstacle, everywhere the vehicle's
clearance is violated, is the complement of a disc, which is not convex:
no quadratic program can state it as a constraint, which is why the
nonlinear formulation reached for a penalty in the first place instead of
a hard bound. The convex program writes a supporting hyperplane instead:
at each predicted step \(k\), a half-space through that step's nominal
position, normal to the line from the obstacle toward it, with a slack
\(\sigma_{k,j} \ge 0\) absorbing whatever penetration the hyperplane still
allows. Because the hyperplane is tangent at the nominal rather than at
the true clearance boundary, it is conservative, and it tightens on every
sequential iterate as the nominal moves. The turn-rate cap in section 4
has the same shape of problem: \(v_{s,k}\sqrt{\kappa(s_k)^2+\varepsilon}
\le \omega_{\max}\) is bilinear in \(v_{s,k}\) and the path parameter, and
becomes the one linear row shown there once \(\kappa\) is frozen at the
nominal arc length (deviation A-30 covers both).

One probe is taken at the vehicle and several more along the reference
ahead of it to find which obstacles are close enough to matter; a barrier
row is then written against every one of those points at every predicted
step, each using the nominal position that step actually lands on rather
than the handful of fixed probe points the earlier implementation reused
across the whole horizon (deviation A-34). That is what keeps the barrier
correct as the vehicle passes an obstacle's nearest point: a barrier
pinned to a fixed probe can start reading as satisfied while the vehicle
is still inside the clearance margin, because the nearest point of a flat
face slides sideways with the vehicle.

The slack's penalty is quadratic, \(w_{\mathrm{obs},k,j}\,\sigma_{k,j}^2\),
with the same forward-cone shape as before but frozen at the nominal
heading and folded into the weight rather than left inside the row — a
cone factor written on the decision variables would multiply two of them
together and undo the convexity, so the sequential loop recovers the
directionality between iterates instead of within one. The quartic
penetration \(p = 4\) of the nonlinear cost becomes this square, so
**`obstacle_barrier_power` is accepted for backward compatibility and no
longer shapes the barrier** (deviation A-30). Deviation A-33 tilts
this normal away from the direction of travel in the joint-space
controller, because an obstacle sitting on the route otherwise leaves the
half-space facing back along it with no lateral gradient to break a
symmetric approach. `PathFollowingMpc` carries no such tilt, and what
follows from that is a stall rather than a collision: an obstacle on the
reference brings the vehicle to a halt at a safe distance and it does not
resume, because the barrier can express stopping short and cannot express
going around. Deviation A-35 records it.

This remains **soft**, not a hard road tube, and it remains a ball around
the nearest reported obstacle point rather than a shape-aware constraint:
[`Occupancy`](../crates/arco-core/src/protocols.rs) reports a nearest
point and a clearance flag, and nothing distinguishes a position inside an
obstacle's body from one in the band around it. A clearance-based cruise
preview (`_preview_cruise_speed`) also scales the cruise cap down before
pinch points enter the horizon.

---

## 6. Online problem and solver

At each control tick, with measured \((X_0, s_0)\), the nonlinear problem
ARCO states is:

\[
\begin{aligned}
\min_{X,U,S,v_s}\quad & J \\
\text{s.t.}\quad
& \text{Dubins + progress dynamics above},\\
& \text{box constraints on } v,\omega,a,\dot\omega,s,\\
& v_{s,k} \in [0,\ \min(v_{\mathrm{cruise}},\ \omega_{\max}/|\kappa|)],\\
& X_0,\, s_0 \text{ fixed from measurement}.
\end{aligned}
\]

ARCO no longer hands this problem to a single nonlinear solver. Instead,
each control tick runs a short **sequential convex programming (SQP)**
loop:

1. Linearize the unicycle dynamics and the contouring, lag, heading and
   turn-rate-cap expressions about a nominal trajectory: the shifted
   previous solution when it is still moving, or the rollout that
   [Progress law](#4-progress-law-linear-reward--curve-limited-cap)
   describes when it is not.
2. Assemble the resulting convex quadratic program, described in
   [Stage cost](#5-stage-cost-what-the-program-minimizes) and [Soft
   obstacle barriers](#soft-obstacle-barriers), and solve it with
   [Clarabel](https://github.com/oxfordcontrol/Clarabel.rs).
3. Take the solution as the next nominal and repeat, stopping once two
   successive iterates agree within `sqp_tolerance` on position, heading
   and arc length, or once `max_sqp_iterations` solves have run (three,
   by default).

Two rows with no counterpart in the nonlinear problem hold the predicted
heading within `trust_heading` (0.5 rad by default) and the predicted arc
length within `trust_arc_length` (5 m by default) of the nominal. They
exist because the affine expansions are written in absolute variables and
nothing else stops a solve from placing its answer where the tangent
plane has stopped describing the true model; the heading carries a radius
because every nonlinear term in the position rows multiplies it, and the
arc length carries one because it is what moves the frame the errors are
measured in.

If the warm-started nominal turns out to describe an infeasible program,
the loop retries once from a fresh rollout nominal before giving up,
which is what tells a genuinely blocked geometry apart from a
linearization point that was merely a poor guess. A step that still finds
no plan, or that measures a non-finite or otherwise invalid state, brakes
instead of commanding a stale one; `MPCStepResult.solver_status` reports
which of `solved`, `solved_inexact`, `invalid_state`, `infeasible`,
`unbounded`, `budget_exhausted` or `numerical` produced the command,
replacing the raw status string the earlier nonlinear solver returned
(deviation A-32). `solved_inexact` means Clarabel converged to its
almost-solved tolerance rather than its exact one on the last solve; it is
not a report of whether the SQP loop itself reached `sqp_tolerance`.

Bit-identical agreement with the earlier nonlinear implementation is
neither achievable nor required here: two solvers can return
different, and equally valid, solutions to the same nonconvex problem.
`FR-MPC-02` in [rust/SPEC.md](rust/SPEC.md) bounds the difference at 20
percent of root-mean-square lateral error instead of asking for an exact
match.

The first predicted state \((v_1, \omega_1)\) is still the command target
for `DubinsVehicle`.

---

## 7. Naming: what this is / is not

| Property | ARCO `DubinsPathFollowingMPC` | Classical racing MPCC |
|----------|-------------------------------|------------------------|
| Path parameter in the model | Yes (\(s\)) | Yes (\(\theta\)) |
| Contour vs lag split | Yes | Yes |
| Nonlinear dynamics | Yes (NMPC, solved as a linearized sequence) | Often yes (NMPCC) |
| Progress law | Free \(v_s\), linear reward, curve-limited cap | Free \(\dot\theta\), linear reward |
| Spatial safety | Soft occupancy barriers | Often **hard corridor / tube** |
| Reference | Planner polyline (may ignore dynamics) | Usually smooth centerline |

So: this is a faithful MPCC with a curve-limited progress cap; it is
**not** yet CMPCC / MPCC++ (hard lane tubes).  Those are natural
extensions once the road corridor from the city mesh is exposed as
constraints.

---

## 8. City race contract

Planners provide **topology** (which road corridor).  The tracker must keep
a dynamically feasible trajectory in that corridor:

1. Brake via the \(v_s \le \omega_{\max}/|\kappa|\) cap before sharp kinks
   (previewed by the 5 s horizon).
2. Keep \(|e_c|\) small with a strict quadratic (city keeps
   \(d_{\mathrm{dz}} = 0\); flat bands chatter).
3. Penalize obstacles so the car does not spend the planner's clearance
   budget into buildings.
4. Keep \(s\) non-decreasing and escape parked equilibria via the
   reference-rollout re-seed.

See `map/city.yml` (`simulator.mpc.*`) and `make_city_vehicle_config()`
for the city Dubins limits.  A full inventory of post–path-planning knobs
is in [control_tracking_params.md](control_tracking_params.md).
Closed-loop quality (finish times, lateral error, footprint-collision
gate) is measured headlessly by `tools/city_tracking_report.py`.

---

## Example usage

```python
from arco.control.mpc import (
    DubinsPathFollowingMPC,
    DubinsVehicleLimits,
    PathFollowingMPCConfig,
)

limits = DubinsVehicleLimits(
    max_speed=16.0,
    min_speed=0.0,
    max_turn_rate=0.70,       # rad/s (~40 deg/s)
    max_acceleration=2.5,
    max_turn_rate_dot=1.57,   # rad/s² (~90 deg/s²)
)
cfg = PathFollowingMPCConfig.create_from_config().with_weight_overrides(
    contour=10.0,
    heading=1.0,
    control=0.3,
    progress=8.0,
    lag=6.0,
    obstacle=120.0,
    contour_deadzone=0.0,
).with_horizon_overrides(step_count=50, dt=0.1)
mpc = DubinsPathFollowingMPC(vehicle_limits=limits, config=cfg)
mpc.set_reference([(0.0, 0.0), (40.0, 0.0), (40.0, 40.0)])
result = mpc.step(pose=(0.0, 0.0, 0.0), speed=12.0, turn_rate=0.0, dt=0.1)
```

Enable in SE(2) races with `simulator.tracker: mpc` in the scenario YAML.

---

*This document reflects the current contouring MPCC in ARCO.  If the stage
cost or progress law changes, update this file in the same PR.*
