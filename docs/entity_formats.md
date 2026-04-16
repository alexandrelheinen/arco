# Entity Description Format Research

## Context

ARCO's issue #131 requires a **canonical, typed entity hierarchy** shared across
all simulations and examples.  Before implementing a custom schema, this note
evaluates existing robotics description formats to determine whether adopting
an off-the-shelf standard or a lightweight derivative is more appropriate than
designing a proprietary one.

---

## Formats Evaluated

### 1. URDF — Unified Robot Description Format

**Origin**: ROS (Robot Operating System), standardised by Open Robotics.  
**Spec**: <https://wiki.ros.org/urdf/XML>  
**Format**: XML.

**Key concepts**

| Element | Role |
|---------|------|
| `<link>` | Rigid body segment: mass, inertia, collision shape, visual mesh |
| `<joint>` | Kinematic connection between two links: type (fixed, revolute, prismatic, continuous, floating, planar), parent, child, axis |
| `<robot>` | Top-level container; tree structure |

**Strengths**
- Widely adopted in ROS ecosystem (Gazebo, MoveIt!, RViz).
- Simple tree structure: easy to parse and reason about.
- Many tools consume it (joint-state publishers, URDF parsers in Python: `urdf_parser_py`, `yourdfpy`).

**Weaknesses for ARCO**
- XML is verbose and not idiomatic Python.
- URDF assumes physical simulation context (masses, inertias, meshes) — most attributes irrelevant for path planning.
- No native support for closed kinematic chains.
- No notion of mobile agents or passive objects.
- Requires a full ROS dependency chain or a custom parser.

**Verdict**: Too heavy. Relevant concepts (links, joints, hierarchy) are worth borrowing; the format itself is not appropriate for ARCO.

---

### 2. SDFormat — Simulation Description Format

**Origin**: Open Robotics (Gazebo / Ignition).  
**Spec**: <http://sdformat.org/spec>  
**Format**: XML.

**Key concepts**

| Element | Role |
|---------|------|
| `<model>` | Top-level rigid-body assembly (analogous to URDF `<robot>`) |
| `<link>` | Rigid body with inertia, collision, visual |
| `<joint>` | Kinematic constraint; extends URDF with ball, screw, gearbox types |
| `<sensor>` | Sensor plugins attached to links |
| `<world>` | Full environment: gravity, physics engine, multiple models, lights |

**Strengths**
- More expressive than URDF: supports closed chains via explicit frames.
- `<world>` scope maps naturally onto an ARCO scene.
- Model includes both the robot and static environment objects.

**Weaknesses for ARCO**
- Even more XML; even more simulation-centric (physics engine parameters, mesh paths, sensor plugins).
- Python bindings are a C++ binding layer (`sdformat13-python`) — heavy.
- Overkill for ARCO's planning-layer focus.

**Verdict**: Useful as conceptual reference for world-scoping and closed chains; not adopted as-is.

---

### 3. MJCF — MuJoCo Model Format

**Origin**: DeepMind / Emo Todorov.  
**Spec**: <https://mujoco.readthedocs.io/en/latest/XMLreference.html>  
**Format**: XML (parsed by `mujoco` Python package).

**Key concepts**

| Element | Role |
|---------|------|
| `<worldbody>` | Root of the kinematic tree |
| `<body>` | Rigid body (position + orientation) |
| `<geom>` | Shape attached to a body (box, sphere, capsule, cylinder, mesh) |
| `<joint>` | DOF between parent and child body |
| `<actuator>` | Motor / position servo / velocity controller |
| `<equality>` | Closed-chain constraints |

**Strengths**
- First-class Python API (`mujoco.MjModel`, `mujoco.MjData`).
- Geometry types map well: sphere ↔ circle/sphere, box ↔ rectangle/cuboid.
- Actuator abstraction directly covers ARCO's CartesianAgent (velocity control + LP filter).
- Passive bodies (no actuators, contact-only) match ARCO's `Object`.

**Weaknesses for ARCO**
- Full physics engine (`mujoco`) is a heavy dependency.
- Format is tightly coupled to MuJoCo's physics model (damping, friction, contacts).
- Geometry must be declared in MuJoCo's constraint solver units.

**Verdict**: Strongest conceptual match. Geometry types (sphere/box) and the
actuated vs. passive body distinction are adopted in ARCO's entity model.
The format itself is not adopted because `mujoco` is too heavy a dependency
for a path-planning library.

---

### 4. Summary and Design Decision

| Criterion | URDF | SDFormat | MJCF | ARCO entity model |
|-----------|------|----------|------|-------------------|
| Python-native | ✗ | ✗ | ✓ (package) | ✓ (dataclasses) |
| Typed | ✗ | ✗ | ✓ | ✓ |
| No external dep. | ✗ | ✗ | ✗ | ✓ |
| Sphere/box geometry | ✓ | ✓ | ✓ | ✓ |
| Actuated vs. passive | ✗ | ✗ | ✓ | ✓ |
| Mobile agents | ✗ | ✗ | partial | ✓ |
| Kinematic chain | ✓ | ✓ | ✓ | ✓ |
| JSON serialisable | ✗ | ✗ | ✗ | ✓ |

**Decision**: Design a **lightweight Python dataclass hierarchy** inspired by
MJCF's geometry taxonomy and actuator/passive-body distinction, and URDF's
link-joint tree structure.  No external parser or physics engine is required.

The schema is described in [ENTITY_MODEL.md](ENTITY_MODEL.md) and implemented
in `src/arco/tools/entity/`.

---

## References

- ROS URDF documentation: <https://wiki.ros.org/urdf>
- SDFormat specification: <http://sdformat.org>
- MuJoCo XML Reference: <https://mujoco.readthedocs.io/en/latest/XMLreference.html>
- Drake multibody: <https://drake.mit.edu/doxygen_cxx/group__multibody.html>
