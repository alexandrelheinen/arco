"""ARCO canonical entity model.

Provides a typed hierarchy of physical entities shared across all ARCO
simulations and examples.

See :doc:`docs/ENTITY_MODEL` for the design rationale and format research.

Public API
----------

Geometry:
    :class:`~arco.tools.entity.base.BoxGeometry`
    :class:`~arco.tools.entity.base.SphereGeometry`
    :func:`~arco.tools.entity.base.geometry_from_dict`

Agents:
    :class:`~arco.tools.entity.agent.DubinsAgent`
    :class:`~arco.tools.entity.agent.CartesianAgent`

Kinematic chain:
    :class:`~arco.tools.entity.kinematic.RevoluteJoint`
    :class:`~arco.tools.entity.kinematic.PrismaticJoint`
    :class:`~arco.tools.entity.kinematic.Link`
    :class:`~arco.tools.entity.kinematic.EndEffector`
    :class:`~arco.tools.entity.kinematic.KinematicChain`

Objects:
    :class:`~arco.tools.entity.object.Object`
"""

from .agent import CartesianAgent, DubinsAgent
from .base import (
    BoxGeometry,
    Entity,
    Geometry,
    SphereGeometry,
    geometry_from_dict,
)
from .kinematic import (
    EndEffector,
    KinematicChain,
    Link,
    PrismaticJoint,
    RevoluteJoint,
)
from .object import Object

__all__ = [
    # geometry
    "BoxGeometry",
    "SphereGeometry",
    "Geometry",
    "geometry_from_dict",
    # base
    "Entity",
    # agents
    "DubinsAgent",
    "CartesianAgent",
    # kinematic chain
    "RevoluteJoint",
    "PrismaticJoint",
    "Link",
    "EndEffector",
    "KinematicChain",
    # objects
    "Object",
]
