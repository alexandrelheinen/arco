# Contouring NMPC (NMPCC-style path following)

ARCO’s SE(2) online tracker `DubinsPathFollowingMPC` is a **nonlinear model
predictive contouring controller** (NMPCC-style): it augments the vehicle
state with a path parameter \(s\), splits tracking into **contouring**
(lateral) and **lag** (progress) costs, and solves a nonlinear program each
control step under Dubins / unicycle dynamics.

It belongs to the same family as Model Predictive Contouring Control (MPCC),
but it is **not** a drop-in of the classical racing MPCC formulation
(free virtual speed \(\dot\theta\), hard corridor tubes).  This note documents
**exactly what ARCO implements**.

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
approximate curvature \(\kappa(s)\) used only for speed limiting.

### Curvature used for \(v_{\mathrm{curve}}\)

At each interior vertex, the turn between consecutive outgoing headings is

\[
\Delta\psi_i = \mathrm{wrap}\bigl(\psi_i - \psi_{i-1}\bigr),\qquad
\Delta s_i = \min(\Delta s_{\mathrm{in}},\,\Delta s_{\mathrm{out}},\, s_{\mathrm{cap}}),
\]

and for non-trivial turns \(\Delta s_i \leftarrow \max(\Delta s_i,\, s_{\mathrm{floor}})\)
with \(s_{\mathrm{cap}} = 20\,\mathrm{m}\) and \(s_{\mathrm{floor}} = 8\,\mathrm{m}\),
then \(\kappa_i = \Delta\psi_i / \Delta s_i\).  A backward max-preview
(\(40\,\mathrm{m}\)) exposes upcoming corner \(\kappa\) on the approach, and
\(|\kappa|\) is clipped to \(\kappa_{\max} = 0.35\,\mathrm{m}^{-1}\) so
short A*/optimizer stubs cannot create Dirac \(\kappa\) that makes
long-horizon IPOPT solves fail (city purple racer stuck at
\(v_{\mathrm{cmd}}=0\)).  A skip-one finite difference
\(\psi_{i+1}-\psi_{i-1}\) can report \(\kappa \approx 0\) on a \(90^\circ\)
L-corner and disable braking — that is why consecutive headings are used.

---

## 2. Prediction model (Dubins / unicycle)

Decision variables over horizon \(N\) with step \(\Delta t\):

| Symbol | Meaning |
|--------|---------|
| \(X_k = (p_x, p_y, \psi, v, \omega)_k\) | pose, speed, yaw rate |
| \(U_k = (a, \dot\omega)_k\) | accel, yaw acceleration |
| \(s_k\) | contouring path parameter (arc length) |

Discrete Euler dynamics (matching `DubinsVehicle.step` saturation semantics):

\[
\begin{aligned}
p_{x,k+1} &= p_{x,k} + v_k \cos\psi_k\,\Delta t,\\
p_{y,k+1} &= p_{y,k} + v_k \sin\psi_k\,\Delta t,\\
\psi_{k+1} &= \psi_k + \omega_k\,\Delta t,\\
v_{k+1} &= v_k + a_k\,\Delta t,\\
\omega_{k+1} &= \omega_k + \dot\omega_k\,\Delta t.
\end{aligned}
\]

Box constraints: \(v \in [v_{\min}, v_{\max}]\),
\(|\omega| \le \omega_{\max}\), \(|a| \le a_{\max}\),
\(|\dot\omega| \le \dot\omega_{\max}\), and \(s_k \in [0, L]\).

---

## 3. Contouring errors

At predicted progress \(s_k\), interpolate
\((x_{\mathrm{ref}}, y_{\mathrm{ref}}, \psi_{\mathrm{ref}}, \kappa)\).

**Contouring (lateral) error** — signed, left positive:

\[
e_{c,k}
=
-(p_{x,k}-x_{\mathrm{ref}})\sin\psi_{\mathrm{ref}}
+(p_{y,k}-y_{\mathrm{ref}})\cos\psi_{\mathrm{ref}}.
\]

**Heading error**:

\[
e_{\psi,k} = \psi_k - \psi_{\mathrm{ref}}(s_k).
\]

Heading stage cost uses a smooth \(2\pi\)-periodic surrogate

\[
\ell_\psi(e_\psi) = \sin^2 e_\psi + (1-\cos e_\psi)^2.
\]

**Deadzone (lane-aware free band).**  Only excess lateral error is penalized:

\[
e_{c,k}^{+}
=
\max\bigl(|e_{c,k}| - d_{\mathrm{dz}},\, 0\bigr),
\qquad
d_{\mathrm{dz}} = \texttt{contour\_deadzone}.
\]

With \(d_{\mathrm{dz}}=0\) this is classical stiff contouring.  City uses a
*small* \(d_{\mathrm{dz}}\) (\(\ll\) road half-width) so corners may widen
**inside** the navigable lane without treating clearance as free space.

---

## 4. Path-parameter dynamics (progress law)

Contouring progress is **coupled** to the vehicle (not a free virtual speed):

\[
s_{k+1}
=
s_k + v_k \,\max\bigl(\cos e_{\psi,k},\, 0\bigr)\,\Delta t.
\]

When \(|e_\psi| > \pi/2\), progress stalls instead of reversing.  The older law
\(\dot s = v\cos e_\psi\) drove \(s\) backward on recovery arcs and produced
the city A\* junction limit cycle.

**Measurement / seed.**  Each `step()` projects the measured pose onto the
path in a local window around the current \(s\), then blends without rewind:

\[
s \leftarrow \max\bigl(0.7\,s + 0.3\,s_{\mathrm{proj}},\, s\bigr).
\]

---

## 5. Speed reference

Curve-limited cruise:

\[
v_{\mathrm{curve}}(s) = \frac{\omega_{\max}}{\max(|\kappa(s)|,\,10^{-3})},\qquad
v_{\mathrm{ref}} = \mathrm{clip}\bigl(\min(v_{\mathrm{cruise}},\, v_{\mathrm{curve}}),\, v_{\min},\, v_{\max}\bigr).
\]

Optionally, path-ahead occupancy clearance further scales \(v_{\mathrm{cruise}}\)
before the NLP (`_preview_cruise_speed`).

Minimum turn radius at a chosen speed: \(R = v / \omega_{\max}\).  For lane
feasibility at a sharp corner, \(v_{\mathrm{ref}}\) must drop until \(R\) fits
inside the road corridor.

---

## 6. Stage cost (what the NLP minimizes)

For \(k = 0,\ldots,N-1\):

\[
\begin{aligned}
J &=
\sum_{k=0}^{N-1}
\Big[
w_c\,(e_{c,k}^{+})^2
+ w_\psi\,\ell_\psi(e_{\psi,k})
+ w_v\,(v_{\mathrm{ref},k}-v_k)^2
+ w_{\mathrm{lag}}\,\max(s_k^{\mathrm{tgt}}-s_k,\,0)^2
+ w_u\,(a_k^2+\dot\omega_k^2)
+ w_{\mathrm{slack}}\,\sigma_k^2
+ J_{\mathrm{obs},k}
\Big] \\
&\quad
+ w_T\bigl((e_{c,N}^{+})^2 + \ell_\psi(e_{\psi,N})\bigr)
+ w_{\mathrm{lag}}\,\max(s_N^{\mathrm{tgt}}-s_N,\,0)^2
+ w_{\mathrm{slack}}\,\sigma_N^2.
\end{aligned}
\]

**Lag schedule** (behind-schedule only; being ahead is free):

\[
s_k^{\mathrm{tgt}} = \min\bigl(s_0 + v_{\mathrm{cruise}}\,k\,\Delta t,\, L\bigr).
\]

Weights map to `PathFollowingMPCConfig` / YAML `simulator.mpc.weights`:

| Weight | Config / YAML key |
|--------|-------------------|
| \(w_c\) | `weight_contour` / `contour` |
| \(w_\psi\) | `weight_heading` / `heading` |
| \(w_v\) | `weight_progress` / `progress` |
| \(w_{\mathrm{lag}}\) | `weight_lag` / `lag` |
| \(w_u\) | `weight_control` / `control` |
| \(w_{\mathrm{obs}}\) | `weight_obstacle` / `obstacle` |
| \(d_{\mathrm{dz}}\) | `contour_deadzone` |

Default global `mpc.yml` is **stiff** (\(d_{\mathrm{dz}}=0\), \(w_{\mathrm{lag}}=0\)).
City overrides are **stiffened lane-aware** (small deadzone ≈ 1.2 m, high
\(w_u\) / \(w_c\) / \(w_{\mathrm{obs}}\), mild lag) so corners may open a
little without zigzag hunting or wall cuts.

### Soft obstacle barriers

When an occupancy map is provided, nearest obstacle samples enter a
directional penalty (stronger in the forward cone):

\[
J_{\mathrm{obs},k}
=
\sum_j
w_{\mathrm{obs}}\,
\Bigl(\max\bigl(\tfrac{c - d_{k,j}}{c},\,0\bigr)\Bigr)^{p}
\bigl(0.2 + 0.8\max(\cos(\psi_k-\beta_{k,j}),\,0)\bigr),
\]

with clearance \(c\), distance \(d_{k,j}\), bearing \(\beta_{k,j}\), and power
\(p=\) `obstacle_barrier_power`.  This is **soft**, not a hard road tube.

---

## 7. Online problem

At each control tick, with measured \((X_0, s_0)\):

\[
\begin{aligned}
\min_{X,U,s,\sigma}\quad & J \\
\text{s.t.}\quad
& \text{Dubins + progress dynamics above},\\
& \text{box constraints on } v,\omega,a,\dot\omega,s,\\
& X_0,\, s_0 \text{ fixed from measurement}.
\end{aligned}
\]

Solved with CasADi `Opti` + IPOPT.  The first input \((a_0,\dot\omega_0)\) is
integrated to produce commands \((v_{\mathrm{cmd}}, \omega_{\mathrm{cmd}})\) for
`DubinsVehicle`.

---

## 8. Naming: what this is / is not

| Property | ARCO `DubinsPathFollowingMPC` | Classical racing MPCC |
|----------|-------------------------------|------------------------|
| Path parameter in the NLP | Yes (\(s\)) | Yes (\(\theta\)) |
| Contour vs lag split | Yes | Yes |
| Nonlinear dynamics in the NLP | Yes (NMPC) | Often yes (NMPCC) |
| Progress law | \(\dot s = v\max(\cos e_\psi,0)\) | Free \(\dot\theta\) (virtual) |
| Spatial safety | Soft occupancy barriers | Often **hard corridor / tube** |
| Reference | Planner polyline (may ignore dynamics) | Usually smooth centerline |

So: **yes, this is an NMPCC-style contouring controller**; it is **not** yet
CMPCC / MPCC++ (hard lane tubes + free virtual progress).  Those are natural
extensions once the road corridor from the city mesh is exposed as
constraints.

---

## 9. City race contract

Planners provide **topology** (which road corridor).  The tracker must keep a
dynamically feasible trajectory in that corridor:

1. Brake via \(\kappa \rightarrow v_{\mathrm{curve}}\) before sharp kinks.
2. Allow small \(e_c\) inside \(d_{\mathrm{dz}}\) (widen the polyline corner).
3. Penalize excess \(|e_c|\) and obstacles so the car does not spend the
   planner’s clearance budget into buildings.
4. Keep \(s\) non-decreasing under large heading error.

See `map/city.yml` (`simulator.mpc.*`) and `make_city_vehicle_config()` for
the soft but lane-viable Dubins limits.

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
    max_turn_rate_dot=0.96,   # rad/s² (~55 deg/s²)
)
cfg = PathFollowingMPCConfig.create_from_config().with_weight_overrides(
    contour=18.0,
    heading=6.0,
    control=4.0,
    progress=3.0,
    lag=2.0,
    obstacle=280.0,
    contour_deadzone=1.2,
)
mpc = DubinsPathFollowingMPC(vehicle_limits=limits, config=cfg)
mpc.set_reference([(0.0, 0.0), (40.0, 0.0), (40.0, 40.0)])
result = mpc.step(pose=(0.0, 0.0, 0.0), speed=12.0, turn_rate=0.0)
```

Enable in SE(2) races with `simulator.tracker: mpc` in the scenario YAML.

---

*This document reflects the current contouring NMPC in ARCO.  If the NLP
cost or progress law changes, update this file in the same PR.*
