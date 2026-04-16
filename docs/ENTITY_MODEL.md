# ARCO Entity Model

## Rationale

See [entity_formats.md](entity_formats.md) for the research note that
justifies the design choices below.

## Hierarchy

```
Entity (abstract base)
├── Agent           — mobile actuated body
│   ├── DubinsAgent — non-holonomic vehicle (heading + speed + turn-rate)
│   └── CartesianAgent — N-DoF Cartesian axes with velocity control
├── Link            — rigid body segment in a kinematic chain
├── Joint           — kinematic connection between two Links
│   ├── RevoluteJoint
│   └── PrismaticJoint
├── EndEffector     — terminal element of a kinematic chain
└── Object          — passive body (no actuators; manipulated by contact)
```

## Geometry

Every entity carries a `geometry` attribute, one of:

| Class | Shape | 2-D analogue | 3-D analogue |
|-------|-------|--------------|--------------|
| `BoxGeometry` | rectangle / cuboid | rectangle | cuboid |
| `SphereGeometry` | circle / sphere | circle | sphere |

Geometry determines how collision and contact are computed.

## State

- `Agent` has a mutable `state` (position + heading + speed for Dubins;
  position + velocity for Cartesian).
- `Object` has a mutable `state` (position + orientation) but no actuators.
- `Link`, `Joint`, `EndEffector` compose into a `KinematicChain`.

## Serialisation

All entities are dataclasses and serialise to/from plain dicts (JSON-safe).
This allows passing entity definitions through ARCO's middleware layer without
coupling to any rendering backend.

## Usage Example

```python
from arco.tools.entity import DubinsAgent, BoxGeometry, Object, SphereGeometry

# Create a Dubins vehicle agent
agent = DubinsAgent(
    name="vehicle",
    geometry=BoxGeometry(half_extents=(1.0, 0.5)),
    state=[0.0, 0.0, 0.0],   # [x, y, heading]
    max_speed=5.0,
    max_turn_rate=1.57,
)

# Create a passive box obstacle
obstacle = Object(
    name="crate",
    geometry=BoxGeometry(half_extents=(0.5, 0.5)),
    state=[3.0, 2.0, 0.0],   # [x, y, heading]
)
```

## Module Layout

```
src/arco/tools/entity/
    __init__.py       ← re-exports public API
    base.py           ← Entity (ABC), Geometry, BoxGeometry, SphereGeometry
    agent.py          ← Agent (ABC), DubinsAgent, CartesianAgent
    kinematic.py      ← Joint, RevoluteJoint, PrismaticJoint, Link, EndEffector, KinematicChain
    object.py         ← Object
```

Tests mirror this layout:

```
tests/tools/entity/
    __init__.py
    test_base.py
    test_agent.py
    test_kinematic.py
    test_object.py
```
