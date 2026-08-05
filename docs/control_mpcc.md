# Contouring NMPC (MPCC path following)

ARCO's SE(2) online tracker `DubinsPathFollowingMPC` is a **nonlinear model
predictive contouring controller** (MPCC): it augments the vehicle state
with a path parameter \(s\) driven by its own **virtual progress speed**
decision variable, splits the position error into **contouring** (lateral)
and **lag** (longitudinal) components, and solves a nonlinear program each
control step under Dubins / unicycle dynamics.

This is the classical Lam / Liniger MPCC structure.  This note documents
**exactly what ARCO implements**, including the design decisions that fix
the historical city-race failure modes (zigzag, junction orbits, parked
stalls at sharp kinks).

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
| NLP / step | `src/arco/control/mpc/path_following.py` |
| Arc-length path, \(\kappa\), projection | `src/arco/control/mpc/reference_path.py` |
| Metrics loop | `src/arco/control/mpc/tracking_loop.py` |
| City / sim factory | `src/arco/simulator/sim/tracking.py` |
| Closed-loop city report | `tools/city_tracking_report.py` |
| Demo (stiff vs lane-aware) | `tools/mpc_progress_first_demo.py` |

Optional dependency: `pip install arco[mpc]` (CasADi + IPOPT).

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

Inside the NLP, the reference is resampled on a uniform arc-length grid
(**2 m resolution**, 200–1500 samples) and looked up through **cubic
B-spline CasADi interpolants**.  Piecewise-linear lookups have
discontinuous gradients exactly at polyline kinks, which stalled IPOPT
(`Maximum_Iterations_Exceeded`) right where tracking is hardest; smooth
splines also round the polyline at the ~2 m sample scale, a desirable
property for a vehicle-feasible target.

### Runway extension

`set_reference` appends one prediction-horizon length of straight
"runway" along the final tangent.  Without it, the progress bounds pinch
\(S\) against the arc-length cap near the goal and the NLP fails on the
last meters of the race.

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
constraints are needed *inside* the NLP.

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
nothing while accelerating away looks expensive over the horizon — IPOPT
then converges to a permanent full stop (the city A\* racer stall).  A
linear reward makes advancement pay everywhere; corner braking stays
feed-forward through the \(v_s\) cap, which the lag term transfers to the
actual vehicle speed.

**Measurement / seed.**  Each `step()` projects the measured pose onto the
path in a local window around the current \(s\) (a global nearest-point
search can flip to another road corridor at junctions) and never rewinds:
\(s \leftarrow \max(s, s_{\mathrm{proj}})\).  Recovery arcs catch up to
\(s\) through the lag cost instead of resetting it.

**Warm start / anti-stall initialization.**  The solver warm-starts from
the shifted previous solution while that solution keeps moving.  If the
warm start advances less than \(\max(1, 0.1\, v_{\mathrm{cruise}} N \Delta t)\)
meters over the horizon (a "parked" solution) while the path ahead allows
motion, the initial guess is rebuilt as a **reference rollout**: poses on
the path, speed accelerating toward the curve-limited cruise.  Seeding
inside the moving basin is what lets IPOPT escape the parked local
minimum at sharp kinks.

---

## 5. Stage cost (what the NLP minimizes)

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
deadzone is configured.

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

When an occupancy map is provided, nearest obstacle samples enter a
directional penalty (stronger in the forward cone).  The cone factor is
the **smooth** projection of the unit obstacle bearing onto the heading
(no `atan2` kinks):

\[
J_{\mathrm{obs},k}
=
\sum_j
w_{\mathrm{obs}}\,
\Bigl(\max\bigl(\tfrac{c - d_{k,j}}{c},\,0\bigr)\Bigr)^{p}
\Bigl(0.2 + 0.8\max\bigl(\tfrac{\cos\psi_k\,\Delta x_{k,j} + \sin\psi_k\,\Delta y_{k,j}}{d_{k,j}},\,0\bigr)\Bigr),
\]

with clearance \(c\), obstacle offset \((\Delta x, \Delta y)\), distance
\(d_{k,j}\), and power \(p =\) `obstacle_barrier_power`.  This is
**soft**, not a hard road tube.  A clearance-based cruise preview
(`_preview_cruise_speed`) additionally scales the cruise cap down before
pinch points enter the horizon.

---

## 6. Online problem

At each control tick, with measured \((X_0, s_0)\):

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

Solved with CasADi `Opti` + IPOPT (acceptable-tolerance early exit
enabled).  The first predicted state \((v_1, \omega_1)\) is the command
target for `DubinsVehicle`.

---

## 7. Naming: what this is / is not

| Property | ARCO `DubinsPathFollowingMPC` | Classical racing MPCC |
|----------|-------------------------------|------------------------|
| Path parameter in the NLP | Yes (\(s\)) | Yes (\(\theta\)) |
| Contour vs lag split | Yes | Yes |
| Nonlinear dynamics in the NLP | Yes (NMPC) | Often yes (NMPCC) |
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

*This document reflects the current contouring MPCC in ARCO.  If the NLP
cost or progress law changes, update this file in the same PR.*
